/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

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
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DNN_ENCODERS__DNN_IMAGE_ENCODER_NODE_HPP_
