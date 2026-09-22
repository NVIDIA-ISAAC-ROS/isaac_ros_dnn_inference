// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_TRITON__TRITON_NODE_HPP_
#define ISAAC_ROS_TRITON__TRITON_NODE_HPP_

#include <triton/core/tritonserver.h>
#include <cuda_runtime.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"
#include "triton_conversions/triton_conversions.hpp"

using StringList = std::vector<std::string>;

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

class TritonNode : public rclcpp::Node
{
public:
  explicit TritonNode(const rclcpp::NodeOptions &);

  ~TritonNode();

  TritonNode(const TritonNode &) = delete;

  TritonNode & operator=(const TritonNode &) = delete;

  static std::vector<triton_conversions::TritonTensor<cuda_buffer_backend::ReadHandle>>
  PrepareInputHandlesForExternalConsumer(
    const isaac_ros_tensor_msgs::msg::TensorList & tensor_list,
    const StringList & input_tensor_names,
    cudaStream_t stream);

private:
  void InputCallback(const isaac_ros_tensor_msgs::msg::TensorList::SharedPtr tensor_list);

  isaac_ros_tensor_msgs::msg::TensorList DoInference(
    const isaac_ros_tensor_msgs::msg::TensorList & input_tensor_list);

  bool InitializeTritonServer();
  void ShutdownTritonServer();
  bool InitializeBindingsMap();

  std::vector<triton_conversions::Tensor> ExecuteInference(
    const isaac_ros_tensor_msgs::msg::TensorList & input_tensor_list);

  std::vector<triton_conversions::Tensor> ProcessInferenceResponse(
    TRITONSERVER_InferenceResponse * response);

  const std::string model_name_;
  const uint32_t max_batch_size_;
  const uint32_t num_concurrent_requests_;
  const StringList model_repository_paths_;
  const bool enable_triton_logging_;
  const bool enable_strict_model_;

  const StringList input_tensor_names_;
  const StringList input_binding_names_;

  const StringList output_tensor_names_;
  const StringList output_binding_names_;

  const int log_level_{0};
  const std::string backend_directory_;

  std::unordered_map<std::string, std::string> input_bindings_map_;
  std::unordered_map<std::string, std::string> output_bindings_map_;

  const int16_t input_queue_size_;
  const int16_t output_queue_size_;

  rclcpp::Subscription<isaac_ros_tensor_msgs::msg::TensorList>::SharedPtr input_sub_;
  rclcpp::Publisher<isaac_ros_tensor_msgs::msg::TensorList>::SharedPtr output_pub_;

  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;

  std::mutex triton_mutex_;
  std::atomic<bool> triton_server_ready_{false};

  std::unique_ptr<void, void (*)(void *)> triton_server_{nullptr, [](void *) {}};

  int64_t request_id_{0};
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TRITON__TRITON_NODE_HPP_
