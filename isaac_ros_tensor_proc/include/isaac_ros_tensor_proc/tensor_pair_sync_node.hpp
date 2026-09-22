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

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/exact_time.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

using TensorList = isaac_ros_tensor_msgs::msg::TensorList;

class TensorPairSyncNode : public rclcpp::Node
{
public:
  explicit TensorPairSyncNode(const rclcpp::NodeOptions & options);
  ~TensorPairSyncNode();

private:
  void SynchronizedCallback(
    const TensorList::ConstSharedPtr & msg1,
    const TensorList::ConstSharedPtr & msg2);

  void UnsynchronizedCallback(
    const TensorList::ConstSharedPtr & msg1,
    const TensorList::ConstSharedPtr & msg2);

  int64_t input_queue_size_;
  int64_t output_queue_size_;
  std::string input_tensor1_name_{};
  std::string input_tensor2_name_{};
  std::string output_tensor1_name_{};
  std::string output_tensor2_name_{};

  ::message_filters::Subscriber<TensorList> tensor1_sub_;
  ::message_filters::Subscriber<TensorList> tensor2_sub_;

  using ExactPolicy = ::message_filters::sync_policies::ExactTime<
    TensorList,
    TensorList>;
  ::message_filters::Synchronizer<ExactPolicy> sync_;

  rclcpp::Publisher<TensorList>::SharedPtr tensor_pub_;
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TENSOR_PROC__TENSOR_PAIR_SYNC_NODE_HPP_
