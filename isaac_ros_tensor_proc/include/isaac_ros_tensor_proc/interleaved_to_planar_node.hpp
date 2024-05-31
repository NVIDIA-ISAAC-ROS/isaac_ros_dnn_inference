// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef ISAAC_ROS_TENSOR_PROC__INTERLEAVED_TO_PLANAR_NODE_HPP_
#define ISAAC_ROS_TENSOR_PROC__INTERLEAVED_TO_PLANAR_NODE_HPP_

#include <string>
#include <vector>

#include "isaac_ros_nitros/nitros_node.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

class InterleavedToPlanarNode : public nitros::NitrosNode
{
public:
  explicit InterleavedToPlanarNode(const rclcpp::NodeOptions & = rclcpp::NodeOptions());
  ~InterleavedToPlanarNode() = default;

  void postLoadGraphCallback() override;

private:
  std::vector<int64_t> input_tensor_shape_;
  int64_t num_blocks_;
  std::string output_tensor_name_;
};
}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia


#endif  // ISAAC_ROS_TENSOR_PROC__INTERLEAVED_TO_PLANAR_NODE_HPP_
