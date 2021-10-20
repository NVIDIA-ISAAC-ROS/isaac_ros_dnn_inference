/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_UNET__UNET_DECODER_NODE_HPP_
#define ISAAC_ROS_UNET__UNET_DECODER_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_nvengine_interfaces/msg/tensor_list.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "rclcpp/rclcpp.hpp"

namespace isaac_ros
{
namespace unet
{

class UNetDecoderNode : public rclcpp::Node
{
public:
  explicit UNetDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~UNetDecoderNode();

private:
/**
 * @brief Callback to decode a tensor list output by a U-Net architecture
 *        and then publish a segmentation mask
 *
 * @param tensor_list_msg The TensorList msg representing the segmentation mask output by U-Net
                          This list should contain only one Tensor, formatted in NHWC format
 */
  void UNetDecoderCallback(
    const isaac_ros_nvengine_interfaces::msg::TensorList::ConstSharedPtr tensor_list_msg);

  // Queue size of subscriber
  int queue_size_;

  // Frame id that the message should be in
  std::string header_frame_id_;

  // The output order of the tensor from U-Net. Note: only NHWC is supported currently.
  std::string tensor_output_order_;

  // The color encoding that the colored segmentation mask should be in
  // This should be either rgb8 or bgr8
  std::string color_segmentation_mask_encoding_;

  // The color palette for the color segmentation mask
  // There should be an element for each class
  // Note: only the first 24 bits are used
  std::vector<int64_t> color_palette_;

  // Subscribes to a Tensor that will be converted to a segmentation mask (image)
  rclcpp::Subscription<isaac_ros_nvengine_interfaces::msg::TensorList>::SharedPtr tensor_list_sub_;

  // Publishes the processed Tensor as a segmentation mask (image)
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_segmentation_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr colored_segmentation_pub_;

  struct UNetDecoderImpl;
  std::unique_ptr<UNetDecoderImpl> impl_;  // Pointer to implementation
};

}  // namespace unet
}  // namespace isaac_ros

#endif  // ISAAC_ROS_UNET__UNET_DECODER_NODE_HPP_
