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

#ifndef ISAAC_ROS_TENSOR_RT__TENSOR_RT_NODE_HPP_
#define ISAAC_ROS_TENSOR_RT__TENSOR_RT_NODE_HPP_

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "NvInferPlugin.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tensorrt_conversions/tensorrt_conversions.hpp"

using StringList = std::vector<std::string>;


namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

class TensorRTNode : public rclcpp::Node
{
public:
  explicit TensorRTNode(const rclcpp::NodeOptions &);

  ~TensorRTNode();

  TensorRTNode(const TensorRTNode &) = delete;

  TensorRTNode & operator=(const TensorRTNode &) = delete;

  // Callback for processing input tensor list
  void InputTensorCallback(
    const isaac_ros_tensor_msgs::msg::TensorList::SharedPtr tensor_list);

  // Method to perform TensorRT inference
  isaac_ros_tensor_msgs::msg::TensorList DoInference(
    const isaac_ros_tensor_msgs::msg::TensorList & input_tensor_list);

  // Initialize TensorRT engine and related components
  void InitializeTensorRTEngine();

  // Load TensorRT engine from file
  void LoadEngineFromFile();

  // Build TensorRT engine from ONNX model
  void BuildEngineFromModel();

  // Setup binding information for inputs and outputs
  void SetupBindings();

private:
  // TensorRT Inference Parameters
  const std::string model_file_path_;
  const std::string engine_file_path_;
  const std::string custom_plugin_lib_;

  // Input tensors
  const StringList input_tensor_names_;
  const StringList input_binding_names_;

  // Output tensors
  const StringList output_tensor_names_;
  const StringList output_binding_names_;

  const bool force_engine_update_;
  const bool verbose_;
  const int64_t max_workspace_size_;
  const int64_t dla_core_;
  const int32_t max_batch_size_;
  const bool enable_fp16_;
  const bool relaxed_dimension_check_;
  const int16_t input_queue_size_;
  const int16_t output_queue_size_;

  rclcpp::Subscription<isaac_ros_tensor_msgs::msg::TensorList>::SharedPtr input_sub_;

  rclcpp::Publisher<isaac_ros_tensor_msgs::msg::TensorList>::SharedPtr output_pub_;

  // TensorRT inference engine
  std::unique_ptr<nvinfer1::ICudaEngine> cuda_engine_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // Binding information
  std::unordered_map<std::string, size_t> output_binding_infos_;
  std::unordered_map<std::string, nvinfer1::Dims> input_binding_dims_;

  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TENSOR_RT__TENSOR_RT_NODE_HPP_
