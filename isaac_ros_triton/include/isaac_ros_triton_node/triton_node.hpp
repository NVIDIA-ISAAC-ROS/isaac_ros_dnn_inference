/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

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
