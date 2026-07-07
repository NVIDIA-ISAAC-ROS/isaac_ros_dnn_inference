// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_dnn_image_encoder/dnn_image_encoder_node.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_handle.hpp"
#include "nvcv/BorderType.h"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

DnnImageEncoderNode::DnnImageEncoderNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("dnn_image_encoder_node", options),
  tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
  input_image_width_(declare_parameter<int32_t>("input_image_width", 0)),
  input_image_height_(declare_parameter<int32_t>("input_image_height", 0)),
  network_image_width_(declare_parameter<int32_t>("network_image_width", 0)),
  network_image_height_(declare_parameter<int32_t>("network_image_height", 0)),
  input_encoding_(declare_parameter<std::string>("input_encoding", "rgb8")),
  enable_padding_(declare_parameter<bool>("enable_padding", true)),
  image_mean_(declare_parameter<std::vector<double>>("image_mean", {0.5, 0.5, 0.5})),
  image_stddev_(declare_parameter<std::vector<double>>("image_stddev", {0.5, 0.5, 0.5})),
  batch_size_(declare_parameter<uint16_t>("batch_size", 1)),
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  input_queue_size_(declare_parameter<int64_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int64_t>("output_queue_size", 10)),
  image_sub_{},
  camera_info_sub_{},
  exact_sync_{ExactPolicy(static_cast<uint32_t>(input_queue_size_)), image_sub_, camera_info_sub_}
{
  if (input_image_width_ == 0) {
    throw std::invalid_argument("[Dnn Image Encoder] Invalid input_image_width");
  }
  if (input_image_height_ == 0) {
    throw std::invalid_argument("[Dnn Image Encoder] Invalid input_image_height");
  }
  if (network_image_width_ == 0) {
    throw std::invalid_argument(
            "[Dnn Image Encoder] Invalid network_image_width, "
            "this needs to be set per the model input requirements.");
  }
  if (network_image_height_ == 0) {
    throw std::invalid_argument(
            "[Dnn Image Encoder] Invalid network_image_height, "
            "this needs to be set per the model input requirements.");
  }

  // Create CUDA stream and memory pool
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("DnnImageEncoderNode");

  const int32_t num_channels_from_encoding =
    sensor_msgs::image_encodings::numChannels(input_encoding_);
  const int64_t min_block_size = static_cast<int64_t>(batch_size_) *
    num_channels_from_encoding * network_image_height_ * network_image_width_ *
    static_cast<int64_t>(sizeof(float));
  memory_pool_block_size_ = std::max(memory_pool_block_size_, min_block_size);

  CHECK_CUDA_ERROR(pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device),
    "[Dnn Image Encoder] Failed to create CUDA memory pool");

  std::vector<float> mean_float(image_mean_.begin(), image_mean_.end());
  std::vector<float> stddev_float(image_stddev_.begin(), image_stddev_.end());
  nvcv::TensorShape::ShapeType shape{nvcv::TensorShape::ShapeType{1, 1, 1,
      static_cast<int64_t>(image_mean_.size())}};
  nvcv::TensorShape tensor_shape{shape, nvcv::TENSOR_NHWC};

  mean_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);
  stddev_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);

  auto mean_data = mean_.exportData<nvcv::TensorDataStridedCuda>();
  auto mean_access = nvcv::TensorDataAccessStridedImagePlanar::Create(*mean_data);
  auto stddev_data = stddev_.exportData<nvcv::TensorDataStridedCuda>();
  auto stddev_access = nvcv::TensorDataAccessStridedImagePlanar::Create(*stddev_data);

  CHECK_CUDA_ERROR(
    cudaMemcpy2DAsync(
      mean_access->sampleData(0), mean_access->rowStride(), mean_float.data(),
      mean_float.size() * sizeof(float), mean_float.size() * sizeof(float), 1,
      cudaMemcpyHostToDevice, *cuda_stream_),
    "cudaMemcpy2D failed");
  CHECK_CUDA_ERROR(
    cudaMemcpy2DAsync(
      stddev_access->sampleData(0), stddev_access->rowStride(), stddev_float.data(),
      stddev_float.size() * sizeof(float), stddev_float.size() * sizeof(float), 1,
      cudaMemcpyHostToDevice, *cuda_stream_),
    "cudaMemcpy2D failed");

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);
  const rmw_qos_profile_t input_qos_profile = input_qos.get_rmw_qos_profile();

  // Create subscribers
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  exact_sync_.registerCallback(
    std::bind(
      &DnnImageEncoderNode::ImageSubCallback, this,
      std::placeholders::_1, std::placeholders::_2));
  image_sub_.subscribe(this, "image", input_qos_profile, sub_options);
  camera_info_sub_.subscribe(this, "camera_info", input_qos_profile, sub_options);

  CalculateResizeAndCropParams();
  const cvcuda_utils::NVCVImageFormat input_format = cvcuda_utils::ToNVCVFormat(input_encoding_);
  resized_tensor_ = nvcv::Tensor(
    batch_size_, {resize_out_img_width_, resize_out_img_height_},
    input_format.format);

  padded_tensor_ = nvcv::Tensor(
    batch_size_, {network_image_width_, network_image_height_},
    input_format.format);

  float_tensor_ = nvcv::Tensor(
    batch_size_, {network_image_width_, network_image_height_},
    input_format.float_format);

  normalized_tensor_ = nvcv::Tensor(
    batch_size_, {network_image_width_, network_image_height_},
    input_format.float_format);

  CHECK_CUDA_ERROR(cudaStreamSynchronize(*cuda_stream_),
    "[Dnn Image Encoder] Failed to synchronize CUDA stream");

  tensor_list_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "tensors", output_qos, pub_options);

  RCLCPP_INFO(get_logger(), "[Dnn Image Encoder] Setup complete");
}

DnnImageEncoderNode::~DnnImageEncoderNode() {}

float DnnImageEncoderNode::GetResizeScalar()
{
  float width_scalar = static_cast<float>(network_image_width_) /
    static_cast<float>(input_image_width_);
  float height_scalar = static_cast<float>(network_image_height_) /
    static_cast<float>(input_image_height_);
  return (width_scalar == height_scalar) ? width_scalar : std::min(width_scalar, height_scalar);
}

void DnnImageEncoderNode::CalculateResizeAndCropParams()
{
  const float resize_factor = enable_padding_ ? GetResizeScalar() : 1.0f;
  resize_out_img_width_ = input_image_width_ * resize_factor;
  resize_out_img_height_ = input_image_height_ * resize_factor;
}

void DnnImageEncoderNode::ImageSubCallback(
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  RCLCPP_DEBUG(rclcpp::get_logger("DnnImageEncoderNode"), "[Dnn Image Encoder] Image received");

  // Get read_handle from the input
  const int32_t num_channels{sensor_msgs::image_encodings::numChannels(image_msg->encoding)};
  const int32_t bytes_per_element =
    sensor_msgs::image_encodings::bitDepth(image_msg->encoding) / CHAR_BIT;
  const cvcuda_utils::NVCVImageFormat format = cvcuda_utils::ToNVCVFormat(image_msg->encoding);
  auto input_handle = cvcuda_utils::WrapCVCUDATensor(
    *image_msg, image_msg->get_read_handle(*cuda_stream_),
    format.format, num_channels, bytes_per_element);

  // Create output Tensor
  nvidia::isaac_ros::nitros::NitrosTensor output_tensor;
  nvidia::isaac_ros::nitros::NitrosTensorShape output_tensor_shape {
    batch_size_,
    num_channels,
    network_image_height_,
    network_image_width_};

  auto output_write_handle = output_tensor.from_pool(
    tensor_name_, pool_, output_tensor_shape, nvidia::isaac_ros::nitros::NitrosDataType::kFloat32,
    *cuda_stream_);
  auto output_handle = cvcuda_utils::WrapCVCUDATensor(
    output_tensor, std::move(output_write_handle), network_image_width_,
    network_image_height_,
    format.planar_float_format, nvcv::TYPE_F32, batch_size_, nvcv::TENSOR_NCHW);

  // Resize
  resize_op_(*cuda_stream_, input_handle.get_tensor(), resized_tensor_, NVCV_INTERP_LINEAR);

  // Pad
  if (enable_padding_) {
    float4 border_value = {0.0f, 0.0f, 0.0f, 0.0f};
    int32_t top = 0;
    int32_t left = 0;
    top = (network_image_height_ - resize_out_img_height_) / 2;
    left = (network_image_width_ - resize_out_img_width_) / 2;
    copy_make_border_op_(*cuda_stream_, resized_tensor_, padded_tensor_, top, left,
      NVCV_BORDER_CONSTANT, border_value);
    convert_op_(*cuda_stream_, padded_tensor_, float_tensor_, 1.0f / 255.f, 0.0f);
  } else {
    convert_op_(*cuda_stream_, resized_tensor_, float_tensor_, 1.0f / 255.f, 0.0f);
  }

  norm_op_(*cuda_stream_, float_tensor_, mean_, stddev_, normalized_tensor_,
    1.0f, 0.0f, 0.0f, CVCUDA_NORMALIZE_SCALE_IS_STDDEV);

  reformat_op_(*cuda_stream_, normalized_tensor_, output_handle.get_tensor());

  // Create ROS header
  std_msgs::msg::Header header;
  header.stamp.sec = image_msg->get_timestamp_sec();
  header.stamp.nanosec = image_msg->get_timestamp_nsec();
  header.frame_id = image_msg->get_frame_id();

  // Create a new tensor list
  auto output_tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(header)
    .AddTensor(output_tensor)
    .Build();

  tensor_list_pub_->publish(output_tensor_list);
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode)
