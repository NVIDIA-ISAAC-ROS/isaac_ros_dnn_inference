// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_DNN_IMAGE_ENCODER__DNN_IMAGE_ENCODER_NODE_HPP_
#define ISAAC_ROS_DNN_IMAGE_ENCODER__DNN_IMAGE_ENCODER_NODE_HPP_

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

class DnnImageEncoderNode : public nitros::NitrosNode
{
public:
  explicit DnnImageEncoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~DnnImageEncoderNode();

  void preLoadGraphCallback() override;
  void postLoadGraphCallback() override;

private:
  void CalculateResizeAndCropParams();
  uint16_t GetResizeScalar();
  // Desired properties of the image
  const uint16_t input_image_width_;
  const uint16_t input_image_height_;
  const uint16_t network_image_width_;
  const uint16_t network_image_height_;
  const bool enable_padding_;
  const std::vector<double> image_mean_;
  const std::vector<double> image_stddev_;
  int64_t num_blocks_;
  uint16_t resize_out_img_width_;
  uint16_t resize_out_img_height_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DNN_IMAGE_ENCODER__DNN_IMAGE_ENCODER_NODE_HPP_
