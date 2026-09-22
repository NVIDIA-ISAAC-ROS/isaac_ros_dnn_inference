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

#include "isaac_ros_tensor_proc/interleaved_to_planar_node.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cvcuda_conversions/cvcuda_conversions.hpp"
#include "isaac_ros_common/qos.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

namespace TensorConv = cvcuda_conversions;

InterleavedToPlanarNode::InterleavedToPlanarNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("interleaved_to_planar_node", options),
  input_tensor_shape_(declare_parameter<std::vector<int64_t>>(
      "input_tensor_shape",
      std::vector<int64_t>())),
  output_tensor_name_(declare_parameter<std::string>("output_tensor_name", "input_tensor")),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")}
{
  if (input_tensor_shape_.empty()) {
    throw std::invalid_argument("[InterleavedToPlanarNode] The input shape is empty!");
  }

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("InterleavedToPlanarNode");

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_sub_ = create_subscription<TensorList>(
    "interleaved_tensor", input_qos_,
    std::bind(&InterleavedToPlanarNode::InterleavedToPlanarCallback, this, std::placeholders::_1),
    sub_options);
  image_pub_ = create_publisher<TensorList>(
    "planar_tensor", output_qos_, pub_options);
}

InterleavedToPlanarNode::~InterleavedToPlanarNode() {}

void InterleavedToPlanarNode::InterleavedToPlanarCallback(
  const TensorList::SharedPtr msg)
{
  if (msg->tensors.empty()) {
    RCLCPP_ERROR(get_logger(), "[InterleavedToPlanarNode] Received empty tensor list");
    return;
  }

  const TensorConv::Tensor & input_tensor = msg->tensors[0];
  nvcv::TensorLayout input_layout;
  if (input_tensor.shape.size() == 3) {
    input_layout = nvcv::TENSOR_HWC;
  } else if (input_tensor.shape.size() == 4) {
    input_layout = nvcv::TENSOR_NHWC;
  } else {
    RCLCPP_ERROR(get_logger(), "[InterleavedToPlanarNode] Invalid input tensor shape!");
    return;
  }

  std::vector<int64_t> output_shape;
  nvcv::TensorLayout output_layout;
  if (input_tensor.shape.size() == 3) {
    output_shape = {input_tensor.shape[2], input_tensor.shape[0], input_tensor.shape[1]};
    output_layout = nvcv::TENSOR_CHW;
  } else {
    output_shape = {input_tensor.shape[0], input_tensor.shape[3],
      input_tensor.shape[1], input_tensor.shape[2]};
    output_layout = nvcv::TENSOR_NCHW;
  }

  TensorConv::Tensor output_tensor;
  {
    auto input_handle = TensorConv::from_input_tensor(input_tensor, *cuda_stream_, input_layout);
    output_tensor = TensorConv::allocate_tensor(
      output_shape, input_tensor.dtype_code, input_tensor.dtype_bits, input_tensor.dtype_lanes);
    auto output_handle = TensorConv::from_output_tensor(
      output_tensor, *cuda_stream_, output_layout);
    reformat_op_(*cuda_stream_, input_handle, output_handle);
  }

  TensorList output_tensor_list;
  output_tensor_list.header = msg->header;
  output_tensor_list.names = {output_tensor_name_};
  output_tensor_list.tensors = {std::move(output_tensor)};
  image_pub_->publish(std::move(output_tensor_list));
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode)
