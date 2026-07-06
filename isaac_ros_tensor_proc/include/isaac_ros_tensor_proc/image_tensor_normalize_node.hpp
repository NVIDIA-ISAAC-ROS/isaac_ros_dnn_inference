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

#ifndef ISAAC_ROS_TENSOR_PROC__IMAGE_TENSOR_NORMALIZE_NODE_HPP_
#define ISAAC_ROS_TENSOR_PROC__IMAGE_TENSOR_NORMALIZE_NODE_HPP_

#include <string>
#include <vector>
#include <memory>

#include "cvcuda/OpNormalize.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "nvcv/Tensor.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{
class ImageTensorNormalizeNode : public rclcpp::Node
{
public:
  explicit ImageTensorNormalizeNode(const rclcpp::NodeOptions & options);
  ~ImageTensorNormalizeNode();

private:
  void ImageTensorNormalizeCallback(
    const nvidia::isaac_ros::nitros::NitrosTensorList::SharedPtr tensor_msg);

  // Parameters
  const std::vector<double> image_mean_;
  const std::vector<double> image_stddev_;
  const std::string input_tensor_name_;
  const std::string output_tensor_name_;
  const int64_t memory_pool_block_size_;
  const int64_t memory_pool_num_blocks_;
  const rclcpp::QoS input_qos_;
  const rclcpp::QoS output_qos_;

  // Subscribers and publishers
  rclcpp::Subscription<nvidia::isaac_ros::nitros::NitrosTensorList>::SharedPtr tensor_list_sub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosTensorList>::SharedPtr tensor_list_pub_;

  nvcv::Tensor mean_;
  nvcv::Tensor stddev_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;

  cvcuda::Normalize normalize_op_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TENSOR_PROC__IMAGE_TENSOR_NORMALIZE_NODE_HPP_
