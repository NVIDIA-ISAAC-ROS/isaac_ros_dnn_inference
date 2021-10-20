/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_unet/unet_decoder_node.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "sensor_msgs/image_encodings.hpp"
#include "std_msgs/msg/header.hpp"

namespace
{
enum TensorOutputOrder
{
  NHWC
};

const std::unordered_map<std::string, int32_t> g_str_to_tensor_output_order({
    {"NHWC", TensorOutputOrder::NHWC}});

const std::unordered_map<std::string, int32_t> g_str_to_channel_size({
    {sensor_msgs::image_encodings::RGB8, 3},
    {sensor_msgs::image_encodings::BGR8, 3},
    {sensor_msgs::image_encodings::MONO8, 1}});

const int32_t kFloat32 = 9;
}   // namespace

namespace isaac_ros
{
namespace unet
{
struct UNetDecoderNode::UNetDecoderImpl
{
  std::string header_frame_id_;
  std::string color_segmentation_mask_encoding_;
  std::vector<int64_t> color_palette_;
  int32_t kTensorHeightIdx;
  int32_t kTensorWidthIdx;
  int32_t kTensorClassIdx;
  int32_t kNumberColorChannels{3};

  void Initialize(
    const std::string & header_frame_id, const std::string & tensor_output_order,
    const std::string & color_segmentation_mask_encoding,
    const std::vector<int64_t> & color_palette)
  {
    header_frame_id_ = header_frame_id;
    color_segmentation_mask_encoding_ = color_segmentation_mask_encoding;
    color_palette_ = color_palette;
    switch (g_str_to_tensor_output_order.at(tensor_output_order)) {
      case TensorOutputOrder::NHWC:
        kTensorHeightIdx = 1;
        kTensorWidthIdx = 2;
        kTensorClassIdx = 3;
        break;
      default:
        throw std::invalid_argument("Received invalid tensor output order!" + tensor_output_order);
        break;
    }
  }

  void OnCallback(
    sensor_msgs::msg::Image & output_raw_segmentation_mask,
    sensor_msgs::msg::Image & output_color_segmentation_mask,
    const isaac_ros_nvengine_interfaces::msg::Tensor & tensor_msg,
    const std_msgs::msg::Header & tensor_header)
  {
    output_raw_segmentation_mask =
      GetEmptyImageMsg(tensor_msg, sensor_msgs::image_encodings::MONO8, tensor_header);
    output_color_segmentation_mask =
      GetEmptyImageMsg(tensor_msg, color_segmentation_mask_encoding_, tensor_header);
    ConvertTensorToSegmentationMask(
      output_raw_segmentation_mask, output_color_segmentation_mask,
      tensor_msg);
  }

  sensor_msgs::msg::Image GetEmptyImageMsg(
    const isaac_ros_nvengine_interfaces::msg::Tensor & tensor_msg,
    const std::string & image_encoding,
    const std_msgs::msg::Header & tensor_header)
  {
    // Create an empty message with the correct dimensions using the tensor
    sensor_msgs::msg::Image image_msg;
    image_msg.encoding = image_encoding;
    image_msg.is_bigendian = false;
    image_msg.header = tensor_header;
    image_msg.header.frame_id = header_frame_id_;
    image_msg.height = tensor_msg.shape.dims[kTensorHeightIdx];
    image_msg.width = tensor_msg.shape.dims[kTensorWidthIdx];
    image_msg.step = sizeof(uint8_t) * image_msg.width * g_str_to_channel_size.at(image_encoding);
    image_msg.data.resize(image_msg.step * image_msg.height);
    return image_msg;
  }

  void ConvertTensorToSegmentationMask(
    sensor_msgs::msg::Image & raw_segmentation_mask,
    sensor_msgs::msg::Image & color_segmentation_mask,
    const isaac_ros_nvengine_interfaces::msg::Tensor & tensor_msg)
  {
    if (tensor_msg.data_type == kFloat32) {
      FillSegmentationMask<float>(raw_segmentation_mask, color_segmentation_mask, tensor_msg);
    } else {
      throw std::runtime_error("Recieved invalid Tensor data! Expected float32!");
    }
  }

  template<typename T>
  void FillSegmentationMask(
    sensor_msgs::msg::Image & raw_segmentation_mask,
    sensor_msgs::msg::Image & color_segmentation_mask,
    const isaac_ros_nvengine_interfaces::msg::Tensor & tensor_msg)
  {
    // Reinterpret the strides (which are in bytes) + data as the relevant data type
    const T * tensor_data = reinterpret_cast<const T *>(tensor_msg.data.data());
    const uint32_t tensor_height_stride = tensor_msg.strides[kTensorHeightIdx] / sizeof(T);
    const uint32_t tensor_width_stride = tensor_msg.strides[kTensorWidthIdx] / sizeof(T);
    const uint32_t tensor_class_stride = tensor_msg.strides[kTensorClassIdx] / sizeof(T);

    // RGB8 vs BGR8 position of encodings
    int red_offset, green_offset, blue_offset;
    if (color_segmentation_mask_encoding_ == sensor_msgs::image_encodings::RGB8) {
      red_offset = 0;
      green_offset = 1;
      blue_offset = 2;
    } else if (color_segmentation_mask_encoding_ == sensor_msgs::image_encodings::BGR8) {
      blue_offset = 0;
      green_offset = 1;
      red_offset = 2;
    } else {
      throw std::runtime_error("Received unsupported color encoding!");
    }

    for (size_t h = 0; h < tensor_msg.shape.dims[kTensorHeightIdx]; ++h) {
      for (size_t w = 0; w < tensor_msg.shape.dims[kTensorWidthIdx]; ++w) {
        // Compute the min and max index. Note: min is inclusive and max is exclusive
        uint32_t effective_tensor_min_idx = h * tensor_height_stride + w * tensor_width_stride;
        uint32_t effective_tensor_max_idx = effective_tensor_min_idx +
          tensor_msg.shape.dims[kTensorClassIdx] * tensor_class_stride;

        // Find the class label
        uint8_t class_id = SearchForHighestProbability<T>(
          tensor_data, effective_tensor_min_idx,
          effective_tensor_max_idx,
          tensor_class_stride);

        // Write the data to the raw segmentation image
        raw_segmentation_mask.data[h * raw_segmentation_mask.step + w] = class_id;

        // Write the appropiate color for the segmentation mask
        color_segmentation_mask.data[h * color_segmentation_mask.step + kNumberColorChannels * w +
          red_offset] = (color_palette_[class_id] >> 16) & 0xFF;
        color_segmentation_mask.data[h * color_segmentation_mask.step + kNumberColorChannels * w +
          green_offset] = (color_palette_[class_id] >> 8) & 0xFF;
        color_segmentation_mask.data[h * color_segmentation_mask.step + kNumberColorChannels * w +
          blue_offset] = color_palette_[class_id] & 0xFF;
      }
    }
  }

  template<typename T>
  uint8_t SearchForHighestProbability(
    const T * tensor_data, uint32_t min_idx, uint32_t max_idx,
    uint32_t stride)
  {
    // Performs a straight forward linear search for the max probability
    T best_element = tensor_data[min_idx];
    uint8_t best_idx = 0;
    uint8_t idx_count = 0;
    for (size_t c = min_idx; c < max_idx; c += stride) {
      if (best_element < tensor_data[c]) {
        best_element = tensor_data[c];
        best_idx = idx_count;
      }
      idx_count++;
    }
    return best_idx;
  }
};

UNetDecoderNode::UNetDecoderNode(const rclcpp::NodeOptions options)
: Node("unet_decoder_node", options),
  // Parameters
  queue_size_(declare_parameter<int>("queue_size", rmw_qos_profile_default.depth)),
  header_frame_id_(declare_parameter<std::string>("frame_id", "")),
  tensor_output_order_(declare_parameter<std::string>("tensor_output_order", "NHWC")),
  color_segmentation_mask_encoding_(declare_parameter<std::string>(
      "color_segmentation_mask_encoding", "rgb8")),
  color_palette_(declare_parameter<std::vector<int64_t>>(
      "color_palette",
      std::vector<int64_t>({}))),
  // Subscribers
  tensor_list_sub_(create_subscription<isaac_ros_nvengine_interfaces::msg::TensorList>(
      "tensor_sub", queue_size_,
      std::bind(&UNetDecoderNode::UNetDecoderCallback, this, std::placeholders::_1))),
  // Publishers
  raw_segmentation_pub_(create_publisher<sensor_msgs::msg::Image>("unet/raw_segmentation_mask", 1)),
  colored_segmentation_pub_(create_publisher<sensor_msgs::msg::Image>(
      "unet/colored_segmentation_mask", 1)),
  // Impl initialization
  impl_(std::make_unique<UNetDecoderImpl>())
{
  // Received empty header frame id
  if (header_frame_id_.empty()) {
    RCLCPP_WARN(get_logger(), "Received empty frame id! Header will be published without one.");
  }

  // Received invalid color segmentation mask encoding
  if (color_segmentation_mask_encoding_ != sensor_msgs::image_encodings::RGB8 &&
    color_segmentation_mask_encoding_ != sensor_msgs::image_encodings::BGR8)
  {
    throw std::runtime_error(
            "Received invalid color segmentation mask encoding: " +
            color_segmentation_mask_encoding_);
  }

  // Received invalid tensor output order
  if (g_str_to_tensor_output_order.find(tensor_output_order_) ==
    g_str_to_tensor_output_order.end())
  {
    throw std::runtime_error("Received invalid tensor output order: " + tensor_output_order_);
  }

  // Received empty color palette
  if (color_palette_.empty()) {
    throw std::invalid_argument(
            "Received empty color palette! Fill this with a 24-bit hex color for each class!");
  }

  impl_->Initialize(
    header_frame_id_, tensor_output_order_, color_segmentation_mask_encoding_,
    color_palette_);
}

void UNetDecoderNode::UNetDecoderCallback(
  const isaac_ros_nvengine_interfaces::msg::TensorList::ConstSharedPtr tensor_list_msg)
{
  if (tensor_list_msg->tensors.size() != 1) {
    RCLCPP_ERROR(
      get_logger(), "Received invalid tensor count! Expected only one tensor. Not processing.");
    return;
  }

  auto tensor_msg = tensor_list_msg->tensors[0];
  sensor_msgs::msg::Image raw_segmentation_mask;
  sensor_msgs::msg::Image color_segmentation_mask;
  try {
    impl_->OnCallback(
      raw_segmentation_mask, color_segmentation_mask, tensor_msg,
      tensor_list_msg->header);
    raw_segmentation_pub_->publish(raw_segmentation_mask);
    colored_segmentation_pub_->publish(color_segmentation_mask);
  } catch (const std::runtime_error & e) {
    RCLCPP_ERROR(get_logger(), e.what());
    return;
  }
}

UNetDecoderNode::~UNetDecoderNode() = default;

}  // namespace unet
}  // namespace isaac_ros

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::unet::UNetDecoderNode)
