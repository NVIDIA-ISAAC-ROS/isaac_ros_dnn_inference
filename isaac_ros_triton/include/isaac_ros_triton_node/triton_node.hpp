// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_TRITON_NODE__TRITON_NODE_HPP_
#define ISAAC_ROS_TRITON_NODE__TRITON_NODE_HPP_

#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

using StringList = std::vector<std::string>;

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

class TritonNode : public nitros::NitrosNode
{
public:
  explicit TritonNode(const rclcpp::NodeOptions &);

  ~TritonNode();

  TritonNode(const TritonNode &) = delete;

  TritonNode & operator=(const TritonNode &) = delete;

  // The callback for submitting parameters to the node's graph
  void postLoadGraphCallback() override;

private:
  // Triton inference parameters
  const std::string model_name_;
  const uint32_t max_batch_size_;
  const uint32_t num_concurrent_requests_;
  const StringList model_repository_paths_;

  // Input tensors
  const StringList input_tensor_names_;
  const StringList input_binding_names_;
  const StringList input_tensor_formats_;

  // Output tensors
  const StringList output_tensor_names_;
  const StringList output_binding_names_;
  const StringList output_tensor_formats_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TRITON_NODE__TRITON_NODE_HPP_
