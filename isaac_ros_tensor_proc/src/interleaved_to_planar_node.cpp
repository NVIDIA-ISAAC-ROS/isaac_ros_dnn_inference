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

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_handle.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{
InterleavedToPlanarNode::InterleavedToPlanarNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("interleaved_to_planar_node", options),
  input_tensor_shape_(declare_parameter<std::vector<int64_t>>(
      "input_tensor_shape",
      std::vector<int64_t>())),
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  output_tensor_name_(declare_parameter<std::string>("output_tensor_name", "input_tensor")),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")}
{
  RCLCPP_DEBUG(get_logger(), "[InterleavedToPlanarNode] Constructor");

  if (input_tensor_shape_.empty()) {
    throw std::invalid_argument("[InterleavedToPlanarNode] The input shape is empty!");
  }

  // Create CUDA resources
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("InterleavedToPlanarNode");

  int64_t min_block_size = static_cast<int64_t>(sizeof(float));
  for (auto d : input_tensor_shape_) {
    min_block_size *= d;
  }
  const int64_t actual_block_size = std::max(memory_pool_block_size_, min_block_size);

  CHECK_CUDA_ERROR(pool_.create(
    static_cast<size_t>(actual_block_size),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device),
    "[InterleavedToPlanarNode] Failed to create memory pool");

  // Create subscribers and publishers
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  image_sub_ = create_subscription<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "interleaved_tensor", input_qos_,
    std::bind(&InterleavedToPlanarNode::InterleavedToPlanarCallback, this, std::placeholders::_1),
    sub_options);
  image_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "planar_tensor", output_qos_, pub_options);

  RCLCPP_DEBUG(get_logger(), "[InterleavedToPlanarNode] Setup complete");
}

InterleavedToPlanarNode::~InterleavedToPlanarNode() {}

void InterleavedToPlanarNode::InterleavedToPlanarCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::SharedPtr msg)
{
  // Get input tensor
  if (msg->get_tensors().empty()) {
    RCLCPP_ERROR(get_logger(), "[InterleavedToPlanarNode] Received empty tensor list");
    return;
  }
  const nvidia::isaac_ros::nitros::NitrosTensor input_tensor = msg->get_tensor(0);
  nvcv::TensorShape::ShapeType input_nvcv_shape;
  nvcv::TensorLayout input_tensor_layout;
  if (input_tensor.shape().rank() == 3) {
    input_nvcv_shape = {input_tensor.shape().dims()[0], input_tensor.shape().dims()[1],
      input_tensor.shape().dims()[2]};
    input_tensor_layout = nvcv::TENSOR_HWC;
  } else if (input_tensor.shape().rank() == 4) {
    input_nvcv_shape = {input_tensor.shape().dims()[0], input_tensor.shape().dims()[1],
      input_tensor.shape().dims()[2], input_tensor.shape().dims()[3]};
    input_tensor_layout = nvcv::TENSOR_NHWC;
  } else {
    RCLCPP_ERROR(get_logger(), "[InterleavedToPlanarNode] Invalid input tensor shape!");
    return;
  }
  nvcv::DataType data_type = cvcuda_utils::ToNVCVDataType(input_tensor.data_type());
  auto input_handle = cvcuda_utils::WrapCVCUDATensor(
    input_tensor, msg->get_read_handle(*cuda_stream_), input_nvcv_shape,
    data_type, input_tensor_layout);

  // Get output tensor
  nvidia::isaac_ros::nitros::NitrosTensor output_tensor;
  nvidia::isaac_ros::nitros::NitrosTensorShape output_tensor_shape;
  nvcv::TensorShape::ShapeType output_nvcv_shape;
  nvcv::TensorLayout output_tensor_layout;
  if (input_tensor.shape().rank() == 3) {
    output_tensor_shape = nvidia::isaac_ros::nitros::NitrosTensorShape{
      input_tensor.shape().dims()[2],
      input_tensor.shape().dims()[0],
      input_tensor.shape().dims()[1]};
    output_nvcv_shape = {input_tensor.shape().dims()[2], input_tensor.shape().dims()[0],
      input_tensor.shape().dims()[1]};
    output_tensor_layout = nvcv::TENSOR_CHW;
  } else if (input_tensor.shape().rank() == 4) {
    output_tensor_shape = nvidia::isaac_ros::nitros::NitrosTensorShape{
      input_tensor.shape().dims()[0], input_tensor.shape().dims()[3],
      input_tensor.shape().dims()[1], input_tensor.shape().dims()[2]};
    output_nvcv_shape = {input_tensor.shape().dims()[0], input_tensor.shape().dims()[3],
      input_tensor.shape().dims()[1], input_tensor.shape().dims()[2]};
    output_tensor_layout = nvcv::TENSOR_NCHW;
  }
  auto output_write_handle = output_tensor.from_pool(
    output_tensor_name_, pool_, output_tensor_shape, input_tensor.data_type(), *cuda_stream_);

  auto output_handle = cvcuda_utils::WrapCVCUDATensor(
    output_tensor, std::move(output_write_handle), output_nvcv_shape,
    data_type, output_tensor_layout);

  // Create NVCV shape for WrapCVCUDATensor
  reformat_op_(*cuda_stream_, input_handle.get_tensor(), output_handle.get_tensor());

  // Create output tensor list
  nvidia::isaac_ros::nitros::NitrosTensorList output_tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(msg->get_header())
    .AddTensor(output_tensor)
    .Build();

  image_pub_->publish(std::move(output_tensor_list));
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode)
