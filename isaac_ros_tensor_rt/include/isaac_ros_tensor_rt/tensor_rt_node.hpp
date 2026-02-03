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

#ifndef ISAAC_ROS_TENSOR_RT__TENSOR_RT_NODE_HPP_
#define ISAAC_ROS_TENSOR_RT__TENSOR_RT_NODE_HPP_

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "NvInferPlugin.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

using StringList = std::vector<std::string>;


namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

class TensorRTNode : public nitros::NitrosNode
{
public:
  explicit TensorRTNode(const rclcpp::NodeOptions &);

  ~TensorRTNode();

  TensorRTNode(const TensorRTNode &) = delete;

  TensorRTNode & operator=(const TensorRTNode &) = delete;

  // The callback for submitting parameters to the node's graph
  void postLoadGraphCallback() override;

private:
  // TensorRT Inference Parameters
  const std::string model_file_path_;
  const std::string engine_file_path_;
  const std::string custom_plugin_lib_;

  // Input tensors
  const StringList input_tensor_names_;
  const StringList input_binding_names_;
  const StringList input_tensor_formats_;

  // Output tensors
  const StringList output_tensor_names_;
  const StringList output_binding_names_;
  const StringList output_tensor_formats_;

  const bool force_engine_update_;
  const bool verbose_;
  const int64_t max_workspace_size_;
  const int64_t dla_core_;
  const int32_t max_batch_size_;
  const bool enable_fp16_;
  const bool relaxed_dimension_check_;
  const int64_t num_blocks_;

  size_t determineMaxTensorBlockSize();
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TENSOR_RT__TENSOR_RT_NODE_HPP_
