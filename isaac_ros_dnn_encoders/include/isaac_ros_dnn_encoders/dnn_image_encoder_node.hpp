// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_DNN_ENCODERS__DNN_IMAGE_ENCODER_NODE_HPP_
#define ISAAC_ROS_DNN_ENCODERS__DNN_IMAGE_ENCODER_NODE_HPP_

#include <string>
#include <vector>

#include "isaac_ros_nitros/nitros_node.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

enum class ResizeMode
{
  kDistort = 0,
  kPad = 1,
  kCrop = 2
};

class DnnImageEncoderNode : public nitros::NitrosNode
{
public:
  explicit DnnImageEncoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~DnnImageEncoderNode();

  void preLoadGraphCallback() override;
  void postLoadGraphCallback() override;

private:
  // Desired properties of the image
  const uint16_t network_image_width_;
  const uint16_t network_image_height_;
  const std::vector<double> image_mean_;
  const std::vector<double> image_stddev_;
  int64_t num_blocks_;
  const ResizeMode resize_mode_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DNN_ENCODERS__DNN_IMAGE_ENCODER_NODE_HPP_
