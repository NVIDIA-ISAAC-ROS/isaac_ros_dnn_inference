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

#include "isaac_ros_tensor_proc/reshape_node.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_handle.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

ReshapeNode::ReshapeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("reshape_node", options),
  input_tensor_layout_(declare_parameter<std::string>(
      "input_tensor_layout", "HWC")),
  output_tensor_layout_(declare_parameter<std::string>(
      "output_tensor_layout", "NHWC")),
  input_tensor_shape_(declare_parameter<std::vector<int64_t>>(
      "input_tensor_shape",
      std::vector<int64_t>{1920, 1200, 3})),
  output_tensor_shape_(declare_parameter<std::vector<int64_t>>(
      "output_tensor_shape",
      std::vector<int64_t>{1, 1920, 1200, 3})),
  output_tensor_name_(declare_parameter<std::string>("output_tensor_name", "output")),
  batch_(declare_parameter<int64_t>("batch", 1)),
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  input_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")),
  output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos"))
{
  if (input_tensor_shape_.empty() || output_tensor_shape_.empty()) {
    throw std::invalid_argument("[ReshapeNode] The input or output tensor shape is empty!");
  }

  const int64_t input_element_count{std::accumulate(
      input_tensor_shape_.begin(),
      input_tensor_shape_.end(),
      1,
      std::multiplies<int64_t>())};

  const int64_t output_element_count{std::accumulate(
      output_tensor_shape_.begin(),
      output_tensor_shape_.end(),
      1,
      std::multiplies<int64_t>())};

  if (input_element_count != output_element_count) {
    throw std::invalid_argument(
      "[ReshapeNode] The input and output tensor element counts do not match!");
  }

  // Initialize memory pool and CUDA stream
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ReshapeNode");

  const int64_t min_block_size = output_element_count * static_cast<int64_t>(sizeof(double));
  const int64_t actual_block_size = std::max(memory_pool_block_size_, min_block_size);

  CHECK_CUDA_ERROR(pool_.create(
    static_cast<size_t>(actual_block_size),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device),
    "[ReshapeNode] Failed to create CUDA memory pool");

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  tensor_sub_ = create_subscription<NitrosTensorList>(
    "tensor", input_qos_, std::bind(&ReshapeNode::tensorSubCallback, this,
    std::placeholders::_1), sub_options);

  tensor_pub_ = create_publisher<NitrosTensorList>(
    "reshaped_tensor", output_qos_, pub_options);

  RCLCPP_INFO(get_logger(), "[ReshapeNode] Setup complete");
}

ReshapeNode::~ReshapeNode() {}

void ReshapeNode::tensorSubCallback(const NitrosTensorList::SharedPtr msg)
{
  RCLCPP_DEBUG(get_logger(), "[ReshapeNode] Tensor list received");

  const nvcv::TensorLayout input_tensor_layout =
    cvcuda_utils::ToNVCVTensorLayout(input_tensor_layout_);
  nvcv::TensorShape::ShapeType input_shape;
  if (input_tensor_layout_ == "HWC" || input_tensor_layout_ == "CHW") {
    input_shape = {input_tensor_shape_[0], input_tensor_shape_[1], input_tensor_shape_[2]};
  } else if (input_tensor_layout_ == "NHWC" || input_tensor_layout_ == "NCHW") {
    input_shape = {input_tensor_shape_[0], input_tensor_shape_[1], input_tensor_shape_[2],
      input_tensor_shape_[3]};
  } else {
    throw std::invalid_argument("[ReshapeNode] Invalid input tensor layout!");
  }
  auto output_tensor_list = std::make_unique<nvidia::isaac_ros::nitros::NitrosTensorList>();
  output_tensor_list->set_timestamp_sec(msg->get_timestamp_sec());
  output_tensor_list->set_timestamp_nsec(msg->get_timestamp_nsec());
  output_tensor_list->set_frame_id(msg->get_frame_id());

  std::vector<uint32_t> dims;
  nvcv::TensorShape::ShapeType output_shape;
  for (size_t j = 0; j < output_tensor_shape_.size(); j++) {
    dims.push_back(static_cast<uint32_t>(output_tensor_shape_[j]));
  }
  if (output_tensor_layout_ == "HWC" || output_tensor_layout_ == "CHW") {
    output_shape = {dims[0], dims[1], dims[2]};
  } else if (output_tensor_layout_ == "NHWC" || output_tensor_layout_ == "NCHW") {
    output_shape = {dims[0], dims[1], dims[2], dims[3]};
  }

  nvidia::isaac_ros::nitros::NitrosTensorShape output_tensor_shape =
    nvidia::isaac_ros::nitros::NitrosTensorShape(dims);
  const nvcv::TensorLayout output_tensor_layout =
    cvcuda_utils::ToNVCVTensorLayout(output_tensor_layout_);

  for (size_t i = 0; i < msg->num_tensors(); i++) {
    // Input tensor
    nvidia::isaac_ros::nitros::NitrosTensor input_tensor = msg->get_tensor(i);
    nvcv::DataType dtype = cvcuda_utils::ToNVCVDataType(input_tensor.data_type());
    auto input_handle = cvcuda_utils::WrapCVCUDATensor(
      input_tensor, msg->get_read_handle(*cuda_stream_, i), input_shape, dtype,
      input_tensor_layout);

    // Output tensor
    nvidia::isaac_ros::nitros::NitrosTensor tensor;
    auto output_write_handle = tensor.from_pool(
      msg->get_tensor(i).get_name(), pool_, output_tensor_shape, msg->get_tensor(i).data_type(),
      *cuda_stream_);
    auto output_handle = cvcuda_utils::WrapCVCUDATensor(
      tensor, std::move(output_write_handle), output_shape, dtype, output_tensor_layout);
    tensor.set_name(output_tensor_name_);
    if (input_shape.size() == output_shape.size()) {
      reformat_op_(*cuda_stream_, input_handle.get_tensor(), output_handle.get_tensor());
    } else if (input_shape.size() + 1 == output_shape.size() && output_shape[0] == 1) {
      auto input_data = msg->get_read_handle(*cuda_stream_, i).get_ptr();
      auto output_data =
        output_handle.get_tensor().exportData<nvcv::TensorDataStridedCuda>()->basePtr();

      cudaMemcpyAsync(output_data, input_data,
        input_tensor.bytes_per_element() * input_tensor.element_count(),
        cudaMemcpyDeviceToDevice, *cuda_stream_);
    } else {
      RCLCPP_DEBUG(get_logger(),
        "[ReshapeNode] Input and output tensor shapes do not match, skipping reformat");
    }
    output_tensor_list->add_tensor(tensor);
  }
  tensor_pub_->publish(std::move(output_tensor_list));
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::ReshapeNode)
