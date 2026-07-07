// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_tensor_proc/normalize_node.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_handle.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "nvcv/TensorDataAccess.hpp"
#include "std_msgs/msg/header.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{
namespace
{
// Map to store the nitros tensor format to nitros image type
const std::unordered_map<std::string, std::string> tensor_to_image_type(
  {{"nitros_tensor_list_nhwc_rgb_f32", "nitros_image_rgb8"},
    {"nitros_tensor_list_nhwc_bgr_f32", "nitros_image_bgr8"}});

constexpr double PER_PIXEL_SCALE = 255.0;
}  // namespace

NormalizeNode::NormalizeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("normalize_node", options),
  image_mean_(declare_parameter<std::vector<double>>(
      "image_mean",
      {0.5, 0.5, 0.5})),
  image_stddev_(declare_parameter<std::vector<double>>(
      "image_stddev",
      {0.5, 0.5, 0.5})),
  input_image_width_(declare_parameter<uint16_t>("input_image_width", 0)),
  input_image_height_(declare_parameter<uint16_t>("input_image_height", 0)),
  output_tensor_name_(declare_parameter<std::string>("output_tensor_name", "image")),
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  input_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")),
  output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos"))
{
  if (image_mean_.size() != 3 || image_stddev_.size() != 3) {
    throw std::invalid_argument(
            "[NormalizeNode] Did not receive 3 image mean channels or 3 image stddev channels");
  }
  if (input_image_width_ == 0) {
    throw std::invalid_argument(
            "[NormalizeNode] Invalid input_image_width");
  }
  if (input_image_height_ == 0) {
    throw std::invalid_argument(
            "[NormalizeNode] Invalid input_image_height");
  }

  // Initialize memory pool and CUDA stream
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("NormalizeNode");

  cudaError_t err = pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device);
  CHECK_CUDA_ERROR(err, "[NormalizeNode] Failed to create CUDA memory pool");

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
    cudaMemcpy2D(
      mean_access->sampleData(0), mean_access->rowStride(), mean_float.data(),
      mean_float.size() * sizeof(float), mean_float.size() * sizeof(float), 1,
      cudaMemcpyHostToDevice),
    "cudaMemcpy2D failed");
  CHECK_CUDA_ERROR(
    cudaMemcpy2D(
      stddev_access->sampleData(0), stddev_access->rowStride(), stddev_float.data(),
      stddev_float.size() * sizeof(float), stddev_float.size() * sizeof(float), 1,
      cudaMemcpyHostToDevice),
    "cudaMemcpy2D failed");

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  image_sub_ = create_subscription<nvidia::isaac_ros::nitros::NitrosImage>(
    "image", input_qos_, std::bind(&NormalizeNode::ImageSubCallback, this,
    std::placeholders::_1), sub_options);

  tensor_list_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "normalized_tensor", output_qos_, pub_options);
  RCLCPP_INFO(get_logger(), "[NormalizeNode] Setup complete");
}

NormalizeNode::~NormalizeNode() {}

void NormalizeNode::ImageSubCallback(const nvidia::isaac_ros::nitros::NitrosImage::SharedPtr msg)
{
  RCLCPP_DEBUG(get_logger(), "[NormalizeNode] Image received");

  auto encoding = msg->encoding;
  try {
    const cvcuda_utils::NVCVImageFormat format = cvcuda_utils::ToNVCVFormat(encoding);
  } catch (const std::invalid_argument & e) {
    RCLCPP_ERROR(get_logger(), "Unsupported image encoding: %s", encoding.c_str());
    throw std::invalid_argument("Unsupported image encoding: " + encoding);
  }

  int32_t num_channels{sensor_msgs::image_encodings::numChannels(encoding)};
  int32_t bytes_per_channel = sensor_msgs::image_encodings::bitDepth(encoding) / CHAR_BIT;
  const cvcuda_utils::NVCVImageFormat input_format = cvcuda_utils::ToNVCVFormat(encoding);
  RCLCPP_DEBUG(get_logger(),
    "[NormalizeNode] Image width: %d, height: %d, num_channels: %d,"
    "bytes_per_channel: %d",
    msg->width, msg->height, num_channels, bytes_per_channel);
  auto input_handle = cvcuda_utils::WrapCVCUDATensor(
    *msg, msg->get_read_handle(*cuda_stream_), input_format.format, num_channels,
    bytes_per_channel);

  // Output tensor
  nvidia::isaac_ros::nitros::NitrosTensor tensor;
  nvidia::isaac_ros::nitros::NitrosTensorShape output_tensor_shape{
    input_image_height_, input_image_width_, num_channels};

  auto output_write_handle = tensor.from_pool(
    output_tensor_name_, pool_, output_tensor_shape,
    nvidia::isaac_ros::nitros::NitrosDataType::kFloat32,
    *cuda_stream_);
  std::vector<int32_t> dims{static_cast<int32_t>(msg->height),
    static_cast<int32_t>(msg->width), static_cast<int32_t>(num_channels)};
  nvcv::TensorShape::ShapeType output_shape{dims[0], dims[1], dims[2]};
  auto output_handle = cvcuda_utils::WrapCVCUDATensor(
    tensor, std::move(output_write_handle), output_shape, nvcv::TYPE_F32, nvcv::TENSOR_HWC);

  nvcv::TensorShape::ShapeType float_shape{static_cast<int32_t>(msg->height),
    static_cast<int32_t>(msg->width), num_channels};
  nvcv::TensorShape float_tensor_shape{float_shape, nvcv::TENSOR_HWC};
  nvcv::Tensor float_tensor = nvcv::Tensor(float_tensor_shape, nvcv::TYPE_F32);
  convert_to_op_(*cuda_stream_, input_handle.get_tensor(), float_tensor, 1.0f / 255.f, 0.0f);
  normalize_op_(*cuda_stream_, float_tensor, mean_, stddev_,
    output_handle.get_tensor(), 1.0f, 0.0f, 0.0f, CVCUDA_NORMALIZE_SCALE_IS_STDDEV);

  std_msgs::msg::Header header;
  header.stamp.sec = msg->get_timestamp_sec();
  header.stamp.nanosec = msg->get_timestamp_nsec();
  header.frame_id = msg->get_frame_id();

  nvidia::isaac_ros::nitros::NitrosTensorList tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(header)
    .AddTensor(tensor)
    .Build();

  tensor_list_pub_->publish(std::move(tensor_list));
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::NormalizeNode)
