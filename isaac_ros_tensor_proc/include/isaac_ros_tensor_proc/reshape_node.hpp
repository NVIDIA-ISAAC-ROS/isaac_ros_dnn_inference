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

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "cvcuda/OpReformat.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{
using nvidia::isaac_ros::nitros::NitrosTensorList;
class ReshapeNode : public rclcpp::Node
{
public:
  explicit ReshapeNode(const rclcpp::NodeOptions & options);
  ~ReshapeNode();

  ReshapeNode & operator=(const ReshapeNode &) = delete;
  ReshapeNode(const ReshapeNode &) = delete;

private:
  void tensorSubCallback(const NitrosTensorList::SharedPtr msg);

  // Parameters
  std::string input_tensor_layout_;
  std::string output_tensor_layout_;
  std::vector<int64_t> input_tensor_shape_;
  std::vector<int64_t> output_tensor_shape_;
  std::string output_tensor_name_;
  size_t batch_;
  const int64_t memory_pool_block_size_;
  const int64_t memory_pool_num_blocks_;
  const rclcpp::QoS input_qos_;
  const rclcpp::QoS output_qos_;

  // Subscribers and publishers
  rclcpp::Subscription<NitrosTensorList>::SharedPtr tensor_sub_;
  rclcpp::Publisher<NitrosTensorList>::SharedPtr tensor_pub_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;

  cvcuda::Reformat reformat_op_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TENSOR_PROC__RESHAPE_NODE_HPP_
