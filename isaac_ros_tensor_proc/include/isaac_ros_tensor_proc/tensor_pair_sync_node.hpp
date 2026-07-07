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

#ifndef ISAAC_ROS_TENSOR_PROC__TENSOR_PAIR_SYNC_NODE_HPP_
#define ISAAC_ROS_TENSOR_PROC__TENSOR_PAIR_SYNC_NODE_HPP_

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_message_filters_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"


namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{
class TensorPairSyncNode : public rclcpp::Node
{
public:
  explicit TensorPairSyncNode(const rclcpp::NodeOptions & options);

  ~TensorPairSyncNode();

private:
  // Callback for synchronized tensor pairs
  void SynchronizedCallback(
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg1,
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg2);

  // Callback for unsynchronized messages
  void UnsynchronizedCallback(
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg1,
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg2);

  // Parameters
  int64_t memory_pool_block_size_;
  int64_t memory_pool_num_blocks_;
  int64_t input_queue_size_;
  int64_t output_queue_size_;
  // Tensor names for input and output
  std::string input_tensor1_name_{};
  std::string input_tensor2_name_{};
  std::string output_tensor1_name_{};
  std::string output_tensor2_name_{};

  ::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosTensorList> tensor1_nitros_sub_;
  ::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosTensorList> tensor2_nitros_sub_;

  // Message filter synchronizer
  using ExactPolicy = ::message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosTensorList,
    nvidia::isaac_ros::nitros::NitrosTensorList>;
  ::message_filters::Synchronizer<ExactPolicy> sync_;

  // Publisher for output NitrosTensorList messages
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosTensorList>::SharedPtr nitros_pub_;

  // CUDA stream for GPU operations
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TENSOR_PROC__TENSOR_PAIR_SYNC_NODE_HPP_
