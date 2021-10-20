/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_TRITON__TRITON_NODE_HPP_
#define ISAAC_ROS_TRITON__TRITON_NODE_HPP_

#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nvengine/gxe_node.hpp"


namespace isaac_ros
{
namespace dnn_inference
{

class TritonNode : public nvengine::GXENode
{
public:
  explicit TritonNode(const rclcpp::NodeOptions &);

  ~TritonNode();

  TritonNode(const TritonNode &) = delete;

  TritonNode & operator=(const TritonNode &) = delete;

private:
  // Triton Inference Parameters
  const int32_t storage_type_;
  const std::string model_name_;
  const uint32_t max_batch_size_;
  const uint32_t num_concurrent_requests_;
  const std::vector<std::string> model_repository_paths_;
  const std::vector<std::string> input_tensor_names_;
  const std::vector<std::string> input_binding_names_;
  const std::vector<std::string> output_tensor_names_;
  const std::vector<std::string> output_binding_names_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros

#endif  // ISAAC_ROS_TRITON__TRITON_NODE_HPP_
