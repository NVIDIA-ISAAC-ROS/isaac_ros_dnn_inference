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
#ifndef ISAAC_ROS_TENSOR_PROC__NORMALIZE_NODE_HPP_
#define ISAAC_ROS_TENSOR_PROC__NORMALIZE_NODE_HPP_

#include <string>
#include <vector>

#include "cvcuda/OpConvertTo.hpp"
#include "cvcuda/OpNormalize.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "nvcv/Tensor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

using TensorList = isaac_ros_tensor_msgs::msg::TensorList;

class NormalizeNode : public rclcpp::Node
{
public:
  explicit NormalizeNode(const rclcpp::NodeOptions & options);
  ~NormalizeNode();

private:
  void ImageSubCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  const std::vector<double> image_mean_;
  const std::vector<double> image_stddev_;
  uint16_t input_image_width_;
  uint16_t input_image_height_;
  std::string output_tensor_name_;
  const rclcpp::QoS input_qos_;
  const rclcpp::QoS output_qos_;

  nvcv::Tensor mean_;
  nvcv::Tensor stddev_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<TensorList>::SharedPtr tensor_list_pub_;

  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  cvcuda::Normalize normalize_op_;
  cvcuda::ConvertTo convert_to_op_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_TENSOR_PROC__NORMALIZE_NODE_HPP_
