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

#include <climits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cvcuda_conversions/cvcuda_conversions.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "nvcv/TensorDataAccess.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

namespace TensorConv = cvcuda_conversions;

NormalizeNode::NormalizeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("normalize_node", options),
  image_mean_(declare_parameter<std::vector<double>>("image_mean", {0.5, 0.5, 0.5})),
  image_stddev_(declare_parameter<std::vector<double>>("image_stddev", {0.5, 0.5, 0.5})),
  input_image_width_(declare_parameter<uint16_t>("input_image_width", 0)),
  input_image_height_(declare_parameter<uint16_t>("input_image_height", 0)),
  output_tensor_name_(declare_parameter<std::string>("output_tensor_name", "image")),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")}
{
  if (image_mean_.size() != 3 || image_stddev_.size() != 3) {
    throw std::invalid_argument(
            "[NormalizeNode] Did not receive 3 image mean channels or 3 image stddev channels");
  }
  if (input_image_width_ == 0) {
    throw std::invalid_argument("[NormalizeNode] Invalid input_image_width");
  }
  if (input_image_height_ == 0) {
    throw std::invalid_argument("[NormalizeNode] Invalid input_image_height");
  }

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("NormalizeNode");

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
  sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "image", input_qos_,
    std::bind(&NormalizeNode::ImageSubCallback, this, std::placeholders::_1), sub_options);
  tensor_list_pub_ = create_publisher<TensorList>(
    "normalized_tensor", output_qos_, pub_options);
  RCLCPP_INFO(get_logger(), "[NormalizeNode] Setup complete");
}

NormalizeNode::~NormalizeNode() {}

void NormalizeNode::ImageSubCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  try {
    cvcuda_utils::ToNVCVFormat(msg->encoding);
  } catch (const std::invalid_argument &) {
    RCLCPP_ERROR(get_logger(), "Unsupported image encoding: %s", msg->encoding.c_str());
    throw;
  }

  const int32_t num_channels = sensor_msgs::image_encodings::numChannels(msg->encoding);
  const std::vector<int64_t> shape{static_cast<int64_t>(msg->height),
    static_cast<int64_t>(msg->width), static_cast<int64_t>(num_channels)};
  constexpr uint8_t kFloatCode = static_cast<uint8_t>(TensorConv::DLDataTypeCode::kFloat);

  TensorConv::Tensor output_tensor;
  {
    auto input_handle = TensorConv::from_input_image_msg(*msg, *cuda_stream_);
    output_tensor = TensorConv::allocate_tensor(shape, kFloatCode, 32);
    auto output_handle = TensorConv::from_output_tensor(
      output_tensor, *cuda_stream_, nvcv::TENSOR_HWC);

    nvcv::TensorShape::ShapeType float_shape{static_cast<int32_t>(msg->height),
      static_cast<int32_t>(msg->width), num_channels};
    nvcv::TensorShape float_tensor_shape{float_shape, nvcv::TENSOR_HWC};
    nvcv::Tensor float_tensor = nvcv::Tensor(float_tensor_shape, nvcv::TYPE_F32);
    convert_to_op_(*cuda_stream_, input_handle, float_tensor, 1.0f / 255.f, 0.0f);
    normalize_op_(*cuda_stream_, float_tensor, mean_, stddev_, output_handle,
      1.0f, 0.0f, 0.0f, CVCUDA_NORMALIZE_SCALE_IS_STDDEV);
  }

  TensorList tensor_list;
  tensor_list.header = msg->header;
  tensor_list.names = {output_tensor_name_};
  tensor_list.tensors = {std::move(output_tensor)};
  tensor_list_pub_->publish(std::move(tensor_list));
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::NormalizeNode)
