// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_tensor_proc/tensor_pair_sync_node.hpp"

#include <cuda_runtime.h>

#include <utility>
#include <vector>

#include "cvcuda_conversions/cvcuda_conversions.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

namespace TensorConv = cvcuda_conversions;

namespace
{

TensorConv::Tensor CopyTensor(const TensorConv::Tensor & src, cudaStream_t stream)
{
  const std::vector<int64_t> shape(src.shape.begin(), src.shape.end());
  TensorConv::Tensor dst = TensorConv::allocate_tensor(
    shape, src.dtype_code, src.dtype_bits, src.dtype_lanes);
  {
    auto input = TensorConv::from_input_tensor(src, stream);
    auto output = TensorConv::from_output_tensor(dst, stream);
    auto input_data = input.exportData<nvcv::TensorDataStridedCuda>();
    auto output_data = output.exportData<nvcv::TensorDataStridedCuda>();
    const size_t size_bytes = TensorConv::num_elements(shape) *
      static_cast<size_t>(TensorConv::bytes_per_element(src.dtype_bits, src.dtype_lanes));
    cudaError_t err = cudaMemcpyAsync(
      output_data->basePtr(), input_data->basePtr(), size_bytes,
      cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(
              std::string("[TensorPairSyncNode] cudaMemcpyAsync failed: ") +
              cudaGetErrorString(err));
    }
  }
  return dst;
}

}  // namespace

TensorPairSyncNode::TensorPairSyncNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("tensor_pair_sync_node", options),
  input_queue_size_{declare_parameter<int64_t>("input_queue_size", 10)},
  output_queue_size_{declare_parameter<int64_t>("output_queue_size", 10)},
  input_tensor1_name_{declare_parameter<std::string>("input_tensor1_name", "tensor1")},
  input_tensor2_name_{declare_parameter<std::string>("input_tensor2_name", "tensor2")},
  output_tensor1_name_{declare_parameter<std::string>("output_tensor1_name", "tensor1")},
  output_tensor2_name_{declare_parameter<std::string>("output_tensor2_name", "tensor2")},
  tensor1_sub_{},
  tensor2_sub_{},
  sync_{ExactPolicy{static_cast<uint32_t>(input_queue_size_)}, tensor1_sub_, tensor2_sub_}
{
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("TensorPairSyncNode");

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  tensor1_sub_.subscribe(this, "tensor1", input_qos, sub_options);
  tensor2_sub_.subscribe(this, "tensor2", input_qos, sub_options);

  sync_.registerCallback(
    std::bind(
      &TensorPairSyncNode::SynchronizedCallback, this,
      std::placeholders::_1, std::placeholders::_2));
  sync_.getPolicy()->registerDropCallback(
    std::bind(
      &TensorPairSyncNode::UnsynchronizedCallback, this,
      std::placeholders::_1, std::placeholders::_2));

  tensor_pub_ = create_publisher<TensorList>(
    "tensor_pub", output_qos, pub_options);
}

TensorPairSyncNode::~TensorPairSyncNode() {}

void TensorPairSyncNode::SynchronizedCallback(
  const TensorList::ConstSharedPtr & msg1,
  const TensorList::ConstSharedPtr & msg2)
{
  if (msg1->header.stamp != msg2->header.stamp) {
    RCLCPP_WARN(get_logger(),
      "Both messages received, but timestamps didn't match, dropping messages!");
    return;
  }

  const TensorConv::Tensor * tensor1 = TensorConv::find_tensor_by_name(*msg1, input_tensor1_name_);
  const TensorConv::Tensor * tensor2 = TensorConv::find_tensor_by_name(*msg2, input_tensor2_name_);
  if (!tensor1 || !tensor2) {
    RCLCPP_ERROR(get_logger(), "[TensorPairSyncNode] Input tensor not found in synchronized pair");
    return;
  }

  TensorList output_list;
  output_list.header = msg1->header;
  output_list.names = {output_tensor1_name_, output_tensor2_name_};
  output_list.tensors = {
    CopyTensor(*tensor1, *cuda_stream_),
    CopyTensor(*tensor2, *cuda_stream_)};
  cudaStreamSynchronize(*cuda_stream_);
  tensor_pub_->publish(std::move(output_list));
}

void TensorPairSyncNode::UnsynchronizedCallback(
  const TensorList::ConstSharedPtr &,
  const TensorList::ConstSharedPtr &)
{
  RCLCPP_WARN(get_logger(), "Received unsynchronized tensor pair - dropping messages");
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::TensorPairSyncNode)
