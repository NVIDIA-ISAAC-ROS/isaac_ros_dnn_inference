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
#ifndef ISAAC_ROS_TENSOR_PROC__RESHAPE_NODE_HPP_
#define ISAAC_ROS_TENSOR_PROC__RESHAPE_NODE_HPP_

#include <string>
#include <vector>

#include "cvcuda/OpReformat.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

using TensorList = isaac_ros_tensor_msgs::msg::TensorList;

class ReshapeNode : public rclcpp::Node
{
public:
  explicit ReshapeNode(const rclcpp::NodeOptions & options);
  ~ReshapeNode();

  ReshapeNode & operator=(const ReshapeNode &) = delete;
  ReshapeNode(const ReshapeNode &) = delete;

private:
  void tensorSubCallback(const TensorList::SharedPtr msg);

  std::string input_tensor_layout_;
  std::string output_tensor_layout_;
  std::vector<int64_t> input_tensor_shape_;
  std::vector<int64_t> output_tensor_shape_;
  std::string output_tensor_name_;
  size_t batch_;
  const rclcpp::QoS input_qos_;
  const rclcpp::QoS output_qos_;

  rclcpp::Subscription<TensorList>::SharedPtr tensor_sub_;
  rclcpp::Publisher<TensorList>::SharedPtr tensor_pub_;

  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  cvcuda::Reformat reformat_op_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TENSOR_PROC__RESHAPE_NODE_HPP_
