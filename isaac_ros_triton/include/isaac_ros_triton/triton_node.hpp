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

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "rclcpp/rclcpp.hpp"

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

private:
  // Callback for input tensor list
  void InputCallback(const nitros::NitrosTensorList::ConstSharedPtr tensor_list);

  // Triton Server functionality
  bool InitializeTritonServer();
  void ShutdownTritonServer();
  bool DoTritonInference(
    const nvidia::isaac_ros::nitros::NitrosTensorList & tensor_list,
    const std_msgs::msg::Header & header);

  bool InitializeBindingsMap();

  // Generic Triton inference implementation
  bool ExecuteInference(
    const nvidia::isaac_ros::nitros::NitrosTensorList & input_tensor_list,
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder & list_builder);

  bool ProcessInferenceResponse(
    TRITONSERVER_InferenceResponse * response,
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder & list_builder,
    std::unordered_map<std::string, std::string> & output_bindings_map,
    std::vector<std::string> output_tensor_names
  );
  // Triton inference parameters
  const std::string model_name_;
  const uint32_t max_batch_size_;
  const uint32_t num_concurrent_requests_;
  const StringList model_repository_paths_;
  const bool enable_triton_logging_;
  const bool enable_strict_model_;

  // Input tensors
  const StringList input_tensor_names_;
  const StringList input_binding_names_;
  const StringList input_tensor_formats_;

  // Output tensors
  const StringList output_tensor_names_;
  const StringList output_binding_names_;
  const StringList output_tensor_formats_;

  // Triton logging level (0 = Error, 1 = Warn, 2 = Info, 3+ = Verbose)
  const int log_level_{0};

  // Optional override for Triton backend directory (empty = auto-detect)
  const std::string backend_directory_;

  // mapping between tensor name and binding name
  std::unordered_map<std::string, std::string> input_bindings_map_;
  std::unordered_map<std::string, std::string> output_bindings_map_;

  const int16_t input_queue_size_;
  const int16_t output_queue_size_;

  // NITROS subscriber for input tensors
  rclcpp::Subscription<nvidia::isaac_ros::nitros::NitrosTensorList>::SharedPtr input_sub_;
  // NITROS publisher for output tensors
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosTensorList>::SharedPtr output_pub_;

  // CUDA resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;

  // Formats
  std::string input_format_;
  std::string output_format_;

  // Triton server state
  std::mutex triton_mutex_;
  bool triton_server_ready_{false};

  // Triton server handles (forward declarations)
  std::unique_ptr<void, void(*)(void *)> triton_server_{nullptr, [](void *){}};

  int64_t request_id_{0};
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TRITON__TRITON_NODE_HPP_
