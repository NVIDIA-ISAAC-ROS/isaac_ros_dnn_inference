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

#include "isaac_ros_tensor_proc/image_to_tensor_node.hpp"

#include <climits>
#include <utility>
#include <vector>

#include "cvcuda_conversions/cvcuda_conversions.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

namespace TensorConv = cvcuda_conversions;

ImageToTensorNode::ImageToTensorNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("image_to_tensor_node", options),
  scale_{declare_parameter<bool>("scale", true)},
  tensor_name_{declare_parameter<std::string>("tensor_name", "tensor")},
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")}
{
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ImageToTensorNode");

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "image", input_qos_,
    std::bind(&ImageToTensorNode::ImageToTensorCallback, this, std::placeholders::_1),
    sub_options);
  tensor_pub_ = create_publisher<TensorList>(
    "tensor", output_qos_, pub_options);
}

void ImageToTensorNode::ImageToTensorCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
  const int32_t num_channels = sensor_msgs::image_encodings::numChannels(img_msg->encoding);
  const std::vector<int64_t> shape{static_cast<int64_t>(img_msg->height),
    static_cast<int64_t>(img_msg->width), static_cast<int64_t>(num_channels)};
  constexpr uint8_t kFloatCode = static_cast<uint8_t>(TensorConv::DLDataTypeCode::kFloat);

  TensorConv::Tensor output_tensor;
  {
    auto input_handle = TensorConv::from_input_image_msg(*img_msg, *cuda_stream_);
    output_tensor = TensorConv::allocate_tensor(shape, kFloatCode, 32);
    auto output_handle = TensorConv::from_output_tensor(
      output_tensor, *cuda_stream_, nvcv::TENSOR_HWC);
    const float scale_factor = scale_ ? 1.0f / 255.0f : 1.0f;
    convert_op_(*cuda_stream_, input_handle, output_handle, scale_factor, 0.0f);
  }

  TensorList tensor_list;
  tensor_list.header = img_msg->header;
  tensor_list.names = {tensor_name_};
  tensor_list.tensors = {std::move(output_tensor)};
  tensor_pub_->publish(std::move(tensor_list));
}

ImageToTensorNode::~ImageToTensorNode() {}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::ImageToTensorNode)
