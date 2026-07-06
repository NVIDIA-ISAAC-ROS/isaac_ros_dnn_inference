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
#ifndef ISAAC_ROS_TENSOR_PROC__INTERLEAVED_TO_PLANAR_NODE_HPP_
#define ISAAC_ROS_TENSOR_PROC__INTERLEAVED_TO_PLANAR_NODE_HPP_

#include <string>
#include <vector>

#include "cvcuda/OpReformat.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

#include "rclcpp/rclcpp.hpp"
#include "nvcv/Tensor.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{
class InterleavedToPlanarNode : public rclcpp::Node
{
public:
  explicit InterleavedToPlanarNode(const rclcpp::NodeOptions & options);
  ~InterleavedToPlanarNode();

private:
  void InterleavedToPlanarCallback(
    const nvidia::isaac_ros::nitros::NitrosTensorList::SharedPtr msg);

  // Parameters
  std::vector<int64_t> input_tensor_shape_;
  const int64_t memory_pool_block_size_;
  const int64_t memory_pool_num_blocks_;
  std::string output_tensor_name_;
  const rclcpp::QoS input_qos_;
  const rclcpp::QoS output_qos_;

  // Subscribers & Publishers
  rclcpp::Subscription<nvidia::isaac_ros::nitros::NitrosTensorList>::SharedPtr image_sub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosTensorList>::SharedPtr image_pub_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;

  cvcuda::Reformat reformat_op_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TENSOR_PROC__INTERLEAVED_TO_PLANAR_NODE_HPP_
