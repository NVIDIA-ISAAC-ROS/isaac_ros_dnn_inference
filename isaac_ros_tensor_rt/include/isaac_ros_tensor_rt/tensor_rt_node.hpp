/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_TENSOR_RT__TENSOR_RT_NODE_HPP_
#define ISAAC_ROS_TENSOR_RT__TENSOR_RT_NODE_HPP_

#include <string>
#include <utility>
#include <vector>
#include <memory>

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
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TENSOR_RT__TENSOR_RT_NODE_HPP_
