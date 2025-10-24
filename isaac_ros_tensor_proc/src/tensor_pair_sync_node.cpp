// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_data_type.hpp"

#include "isaac_ros_common/cuda_stream.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

TensorPairSyncNode::TensorPairSyncNode(const rclcpp::NodeOptions options)
: rclcpp::Node("tensor_pair_sync_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  tensor1_nitros_sub_{},
  tensor2_nitros_sub_{},
  sync_{ExactPolicy{3}, tensor1_nitros_sub_, tensor2_nitros_sub_},
  nitros_pub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosTensorList>>(
      this,
      "tensor_pub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_t::supported_type_name,
      nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, output_qos_)},
  input_tensor1_name_{declare_parameter<std::string>(
      "input_tensor1_name",
      "tensor1")},
  input_tensor2_name_{declare_parameter<std::string>(
      "input_tensor2_name",
      "tensor2")},
  output_tensor1_name_{declare_parameter<std::string>(
      "output_tensor1_name",
      "tensor1")},
  output_tensor2_name_{declare_parameter<std::string>(
      "output_tensor2_name",
      "tensor2")}
{
  CHECK_CUDA_ERROR(
    ::nvidia::isaac_ros::common::initNamedCudaStream(
      stream_, "isaac_ros_tensor_pair_sync_node"),
    "Error initializing CUDA stream");

  tensor1_nitros_sub_.subscribe(this, "tensor1");
  tensor2_nitros_sub_.subscribe(this, "tensor2");

  // Register synchronized callback
  sync_.registerCallback(
    std::bind(
      &TensorPairSyncNode::SynchronizedCallback, this,
      std::placeholders::_1, std::placeholders::_2));

  // Register drop callback for unsynchronized messages
  sync_.getPolicy()->registerDropCallback(
    std::bind(
      &TensorPairSyncNode::UnsynchronizedCallback, this,
      std::placeholders::_1, std::placeholders::_2));
}

TensorPairSyncNode::~TensorPairSyncNode()
{
  cudaStreamDestroy(stream_);
}

void TensorPairSyncNode::SynchronizedCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg1,
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg2)
{
  RCLCPP_DEBUG(this->get_logger(), "Processing synchronized tensor pair!");

  auto msg1_view = nvidia::isaac_ros::nitros::NitrosTensorListView(*msg1);
  auto msg2_view = nvidia::isaac_ros::nitros::NitrosTensorListView(*msg2);

  if (msg1_view.GetTimestampSeconds() != msg2_view.GetTimestampSeconds() ||
    msg1_view.GetTimestampNanoseconds() != msg2_view.GetTimestampNanoseconds())
  {
    RCLCPP_WARN(this->get_logger(),
      "Both messages received, but timestamps didn't match, dropping messages!");
    return;
  }

  // Forward header from input message
  std_msgs::msg::Header header{};
  header.stamp.sec = msg1_view.GetTimestampSeconds();
  header.stamp.nanosec = msg1_view.GetTimestampNanoseconds();
  header.frame_id = msg1_view.GetFrameId();

  // Get tensors
  auto tensor1 = msg1_view.GetNamedTensor(input_tensor1_name_);
  auto tensor2 = msg2_view.GetNamedTensor(input_tensor2_name_);

  // Convert PrimitiveType to NitrosDataType (they have similar enum values)
  auto tensor1_data_type = static_cast<nvidia::isaac_ros::nitros::NitrosDataType>(
    static_cast<int>(tensor1.GetElementType()));
  auto tensor2_data_type = static_cast<nvidia::isaac_ros::nitros::NitrosDataType>(
    static_cast<int>(tensor2.GetElementType()));

  // Process tensor1
  void * tensor1_output_buffer;
  cudaMallocAsync(&tensor1_output_buffer, tensor1.GetTensorSize(), stream_);
  cudaMemcpyAsync(
    tensor1_output_buffer, tensor1.GetBuffer(),
    tensor1.GetTensorSize(), cudaMemcpyDefault, stream_);

  // Process tensor2
  void * tensor2_output_buffer;
  cudaMallocAsync(&tensor2_output_buffer, tensor2.GetTensorSize(), stream_);
  cudaMemcpyAsync(
    tensor2_output_buffer, tensor2.GetBuffer(),
    tensor2.GetTensorSize(), cudaMemcpyDefault, stream_);

  cudaStreamSynchronize(stream_);

  // Create output tensor list with both processed tensors using their original datatypes
  auto output_tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(header)
    .AddTensor(
      output_tensor1_name_,
    (
      nvidia::isaac_ros::nitros::NitrosTensorBuilder()
      .WithShape(tensor1.GetShape())
      .WithDataType(tensor1_data_type)
      .WithData(tensor1_output_buffer)
      .Build()
    )
    )
    .AddTensor(
      output_tensor2_name_,
    (
      nvidia::isaac_ros::nitros::NitrosTensorBuilder()
      .WithShape(tensor2.GetShape())
      .WithDataType(tensor2_data_type)
      .WithData(tensor2_output_buffer)
      .Build()
    )
    )
    .Build();

  nitros_pub_->publish(output_tensor_list);
}

void TensorPairSyncNode::UnsynchronizedCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg1,
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg2)
{
  RCLCPP_WARN(this->get_logger(), "Received unsynchronized tensor pair - dropping messages");
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::TensorPairSyncNode)
