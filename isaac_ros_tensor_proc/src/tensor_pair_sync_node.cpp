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

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_data_type.hpp"

#include "isaac_ros_common/cuda_stream.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

TensorPairSyncNode::TensorPairSyncNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("tensor_pair_sync_node", options),
  memory_pool_block_size_{declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)},
  memory_pool_num_blocks_{declare_parameter<int64_t>("memory_pool_num_blocks", 40)},
  input_queue_size_{declare_parameter<int64_t>("input_queue_size", 10)},
  output_queue_size_{declare_parameter<int64_t>("output_queue_size", 10)},
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
    "tensor2")},
  tensor1_nitros_sub_{},
  tensor2_nitros_sub_{},
  sync_{ExactPolicy{static_cast<uint32_t>(input_queue_size_)}, tensor1_nitros_sub_,
    tensor2_nitros_sub_}
{
  RCLCPP_DEBUG(get_logger(), "[TensorPairSyncNode] Constructor");

  // Create CUDA resources
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("TensorPairSyncNode");

  CHECK_CUDA_ERROR(pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device),
    "[TensorPairSyncNode] Failed to create CUDA memory pool");

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);
  const rmw_qos_profile_t rmw_qos_profile = input_qos.get_rmw_qos_profile();

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  tensor1_nitros_sub_.subscribe(this, "tensor1", rmw_qos_profile, sub_options);
  tensor2_nitros_sub_.subscribe(this, "tensor2", rmw_qos_profile, sub_options);

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

  nitros_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "tensor_pub", output_qos, pub_options);
  RCLCPP_DEBUG(get_logger(), "[TensorPairSyncNode] Setup complete");
}

TensorPairSyncNode::~TensorPairSyncNode() {}

void TensorPairSyncNode::SynchronizedCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg1,
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg2)
{
  RCLCPP_DEBUG(this->get_logger(), "Processing synchronized tensor pair!");

  if (msg1->get_timestamp_sec() != msg2->get_timestamp_sec() ||
    msg1->get_timestamp_nsec() != msg2->get_timestamp_nsec())
  {
    RCLCPP_WARN(this->get_logger(),
      "Both messages received, but timestamps didn't match, dropping messages!");
    return;
  }

  // Forward header from input message
  std_msgs::msg::Header header{};
  header.stamp.sec = msg1->get_timestamp_sec();
  header.stamp.nanosec = msg1->get_timestamp_nsec();
  header.frame_id = msg1->get_frame_id();

  // Get tensors
  auto tensor1 = msg1->get_tensor(0);
  auto tensor2 = msg2->get_tensor(0);

  // Convert PrimitiveType to NitrosDataType (they have similar enum values)
  auto tensor1_data_type = tensor1.data_type();
  auto tensor2_data_type = tensor2.data_type();

  // Process tensor1
  void * tensor1_output_buffer;
  size_t tensor1_data_size = tensor1.bytes_per_element() * tensor1.element_count();
  cudaMallocAsync(&tensor1_output_buffer, tensor1_data_size, *cuda_stream_);
  cudaMemcpyAsync(
    tensor1_output_buffer, tensor1.get_read_handle(*cuda_stream_).get_ptr(),
    tensor1_data_size, cudaMemcpyDefault, *cuda_stream_);

  // Process tensor2
  void * tensor2_output_buffer;
  size_t tensor2_data_size = tensor2.bytes_per_element() * tensor2.element_count();

  cudaMallocAsync(&tensor2_output_buffer, tensor2_data_size, *cuda_stream_);
  cudaMemcpyAsync(
    tensor2_output_buffer, tensor2.get_read_handle(*cuda_stream_).get_ptr(),
    tensor2_data_size, cudaMemcpyDefault, *cuda_stream_);

  cudaStreamSynchronize(*cuda_stream_);

  // Create output tensor list with both processed tensors using their original datatypes
  auto output_tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(header)
    .AddTensor(
      nvidia::isaac_ros::nitros::NitrosTensorBuilder()
    .WithName(output_tensor1_name_)
    .WithShape(tensor1.shape())
    .WithDataType(tensor1_data_type)
    .WithData(tensor1_output_buffer)
    .WithReleaseCallback([tensor1_output_buffer, stream = *cuda_stream_]() {
      cudaFreeAsync(tensor1_output_buffer, stream);
    })
    .Build()
    )
    .AddTensor(
      nvidia::isaac_ros::nitros::NitrosTensorBuilder()
    .WithName(output_tensor2_name_)
    .WithShape(tensor2.shape())
    .WithDataType(tensor2_data_type)
    .WithData(tensor2_output_buffer)
    .WithReleaseCallback([tensor2_output_buffer, stream = *cuda_stream_]() {
      cudaFreeAsync(tensor2_output_buffer, stream);
  })
    .Build()
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
