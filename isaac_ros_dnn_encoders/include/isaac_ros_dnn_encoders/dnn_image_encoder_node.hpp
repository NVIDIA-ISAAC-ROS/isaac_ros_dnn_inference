/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_DNN_ENCODERS__DNN_IMAGE_ENCODER_NODE_HPP_
#define ISAAC_ROS_DNN_ENCODERS__DNN_IMAGE_ENCODER_NODE_HPP_

#include <memory>
#include <string>

#include "image_transport/image_transport.hpp"
#include "isaac_ros_nvengine_interfaces/msg/tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"

namespace isaac_ros
{
namespace dnn_inference
{

class DnnImageEncoderNode : public rclcpp::Node
{
public:
  explicit DnnImageEncoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~DnnImageEncoderNode();

private:
/**
 * @brief Callback to preprocess an image for the DNN inference node and then publish
 *         the processed Tensor for the DNN inference node to use
 *
 * @param image_msg The image message received
 */
  void DnnImageEncoderCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr image_msg);

  // Desired properties of the image
  const int network_image_width_;
  const int network_image_height_;
  const std::string network_image_encoding_;

  // Name of the published Tensor message
  const std::string tensor_name_;

  // Method to normalize image. Supported types are "unit_scaling" (range is [0, 1]),
  // and "positive_negative" (range is [-1, 1]) and "none" for no normalization
  const std::string network_normalization_type_;

  // Image subscriber
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  // Publisher for outputting the processed image as a TensorList (containing one Tensor)
  rclcpp::Publisher<isaac_ros_nvengine_interfaces::msg::TensorList>::SharedPtr tensor_pub_;

  struct DnnImageEncoderImpl;
  std::unique_ptr<DnnImageEncoderImpl> impl_;  // Pointer to implementation
};

}  // namespace dnn_inference
}  // namespace isaac_ros

#endif  // ISAAC_ROS_DNN_ENCODERS__DNN_IMAGE_ENCODER_NODE_HPP_
