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

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "cvcuda_conversions/cvcuda_conversions.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

namespace TensorConv = cvcuda_conversions;

ReshapeNode::ReshapeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("reshape_node", options),
  input_tensor_layout_(declare_parameter<std::string>("input_tensor_layout", "HWC")),
  output_tensor_layout_(declare_parameter<std::string>("output_tensor_layout", "NHWC")),
  input_tensor_shape_(declare_parameter<std::vector<int64_t>>(
      "input_tensor_shape",
      std::vector<int64_t>{1920, 1200, 3})),
  output_tensor_shape_(declare_parameter<std::vector<int64_t>>(
      "output_tensor_shape",
      std::vector<int64_t>{1, 1920, 1200, 3})),
  output_tensor_name_(declare_parameter<std::string>("output_tensor_name", "output")),
  batch_(declare_parameter<int64_t>("batch", 1)),
  input_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")),
  output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos"))
{
  if (input_tensor_shape_.empty() || output_tensor_shape_.empty()) {
    throw std::invalid_argument("[ReshapeNode] The input or output tensor shape is empty!");
  }

  const int64_t input_element_count = std::accumulate(
    input_tensor_shape_.begin(), input_tensor_shape_.end(), int64_t{1}, std::multiplies<int64_t>());
  const int64_t output_element_count = std::accumulate(
    output_tensor_shape_.begin(), output_tensor_shape_.end(), int64_t{1},
    std::multiplies<int64_t>());

  if (input_element_count != output_element_count) {
    throw std::invalid_argument(
            "[ReshapeNode] The input and output tensor element counts do not match!");
  }

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ReshapeNode");

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  tensor_sub_ = create_subscription<TensorList>(
    "tensor", input_qos_,
    std::bind(&ReshapeNode::tensorSubCallback, this, std::placeholders::_1), sub_options);
  tensor_pub_ = create_publisher<TensorList>(
    "reshaped_tensor", output_qos_, pub_options);

  RCLCPP_INFO(get_logger(), "[ReshapeNode] Setup complete");
}

ReshapeNode::~ReshapeNode() {}

void ReshapeNode::tensorSubCallback(const TensorList::SharedPtr msg)
{
  const nvcv::TensorLayout input_layout =
    cvcuda_utils::ToNVCVTensorLayout(input_tensor_layout_);
  const nvcv::TensorLayout output_layout =
    cvcuda_utils::ToNVCVTensorLayout(output_tensor_layout_);

  // Match the pre-migration node: interpret buffers using the configured
  // shape/layout parameters, not the message's declared shape. Downstream
  // graphs (e.g. dnn_image_encoder) rely on this to add a batch dim via memcpy
  // even when upstream already published a different rank.
  const size_t input_rank = input_tensor_shape_.size();
  const size_t output_rank = output_tensor_shape_.size();

  TensorList output_tensor_list;
  output_tensor_list.header = msg->header;
  output_tensor_list.names.reserve(msg->tensors.size());
  output_tensor_list.tensors.reserve(msg->tensors.size());

  for (size_t i = 0; i < msg->tensors.size(); ++i) {
    const TensorConv::Tensor & input_tensor = msg->tensors[i];
    const size_t msg_elems = TensorConv::num_elements(
      std::vector<int64_t>(input_tensor.shape.begin(), input_tensor.shape.end()));
    const size_t cfg_elems = TensorConv::num_elements(input_tensor_shape_);
    if (msg_elems != cfg_elems) {
      RCLCPP_ERROR(
        get_logger(),
        "[ReshapeNode] Input tensor element count (%zu) does not match configured "
        "input_tensor_shape (%zu); skipping",
        msg_elems, cfg_elems);
      continue;
    }

    TensorConv::Tensor output_tensor;
    try {
      auto input_handle = TensorConv::from_input_tensor(
        input_tensor, *cuda_stream_, input_layout, input_tensor_shape_);
      output_tensor = TensorConv::allocate_tensor(
        output_tensor_shape_, input_tensor.dtype_code, input_tensor.dtype_bits,
        input_tensor.dtype_lanes);
      auto output_handle = TensorConv::from_output_tensor(
        output_tensor, *cuda_stream_, output_layout);

      if (input_rank == output_rank) {
        reformat_op_(*cuda_stream_, input_handle, output_handle);
      } else if (input_rank + 1 == output_rank && output_tensor_shape_[0] == 1) {
        const size_t size_bytes = cfg_elems * static_cast<size_t>(
          TensorConv::bytes_per_element(input_tensor.dtype_bits, input_tensor.dtype_lanes));
        auto input_data = input_handle.exportData<nvcv::TensorDataStridedCuda>();
        auto output_data = output_handle.exportData<nvcv::TensorDataStridedCuda>();
        cudaError_t err = cudaMemcpyAsync(
          output_data->basePtr(), input_data->basePtr(), size_bytes,
          cudaMemcpyDeviceToDevice, *cuda_stream_);
        if (err != cudaSuccess) {
          RCLCPP_ERROR(
            get_logger(), "[ReshapeNode] cudaMemcpyAsync failed: %s",
            cudaGetErrorString(err));
          continue;
        }
      } else {
        RCLCPP_DEBUG(
          get_logger(),
          "[ReshapeNode] Input and output tensor shapes do not match, skipping reformat");
        continue;
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "[ReshapeNode] Failed to reshape tensor: %s", e.what());
      continue;
    }

    output_tensor_list.names.push_back(output_tensor_name_);
    output_tensor_list.tensors.push_back(std::move(output_tensor));
  }

  if (!output_tensor_list.tensors.empty()) {
    tensor_pub_->publish(std::move(output_tensor_list));
  }
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::ReshapeNode)
