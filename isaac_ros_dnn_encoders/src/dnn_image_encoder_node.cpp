/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_dnn_encoders/dnn_image_encoder_node.hpp"

#include <memory>
#include <string>
#include <unordered_map>

#include "cv_bridge/cv_bridge.h"
#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"

namespace
{
enum NormalizationTypes
{
  kNone,
  kUnitScaling,
  kPositiveNegative
};

const std::unordered_map<std::string, int32_t> g_str_to_normalization_type({
    {"none", NormalizationTypes::kNone},
    {"unit_scaling", NormalizationTypes::kUnitScaling},
    {"positive_negative", NormalizationTypes::kPositiveNegative}});

const std::unordered_map<std::string, std::string> g_str_to_image_encoding({
    {"rgb8", sensor_msgs::image_encodings::RGB8},
    {"bgr8", sensor_msgs::image_encodings::BGR8},
    {"mono8", sensor_msgs::image_encodings::MONO8}});

const int32_t kFloat32 = 9;  // float data type defined in ROS Tensor message
}  // namespace

namespace isaac_ros
{
namespace dnn_inference
{

struct DnnImageEncoderNode::DnnImageEncoderImpl
{
  DnnImageEncoderNode * node;
  int image_width_;
  int image_height_;
  std::string image_encoding_;
  std::string normalization_type_;
  std::string tensor_name_;

  void Initialize(DnnImageEncoderNode * encoder_node)
  {
    node = encoder_node;
    image_width_ = node->network_image_width_;
    image_height_ = node->network_image_height_;
    image_encoding_ = node->network_image_encoding_;
    normalization_type_ = node->network_normalization_type_;
    tensor_name_ = node->tensor_name_;
  }

  isaac_ros_nvengine_interfaces::msg::TensorList OnCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg)
  {
    // Cv bridge conversion and convert the color space
    cv_bridge::CvImagePtr image_ptr;
    image_ptr = cv_bridge::toCvCopy(image_msg, g_str_to_image_encoding.at(image_encoding_));

    // Resize the image to the user specified dimensions
    cv::Mat image_resized;
    cv::resize(image_ptr->image, image_resized, cv::Size(image_width_, image_height_));

    // Normalize tensor depending on normalization type required
    switch (g_str_to_normalization_type.at(normalization_type_)) {
      case NormalizationTypes::kUnitScaling:
        image_resized.convertTo(image_resized, CV_32F, 1.0f / 255.0f);
        break;
      case NormalizationTypes::kPositiveNegative:
        image_resized.convertTo(image_resized, CV_32F, 2.0f / 255.0f, -1.0f);
        break;
      default:
        image_resized.convertTo(image_resized, CV_32F);
    }

    // Convert to NCHW
    cv::Mat cv_tensor = cv::dnn::blobFromImage(image_resized);

    // Convert CV matrix to ROS2 tensor message
    auto tensor_list_msg = isaac_ros_nvengine_interfaces::msg::TensorList();
    tensor_list_msg.header = image_msg->header;
    tensor_list_msg.tensors.push_back(isaac_ros_nvengine_interfaces::msg::Tensor());
    auto & tensor = tensor_list_msg.tensors[0];

    tensor.name = tensor_name_;
    tensor.data_type = kFloat32;

    // Copy the dimensions of the tensor
    tensor.shape.rank = cv_tensor.dims;
    for (int i = 0; i < cv_tensor.dims; ++i) {
      tensor.shape.dims.push_back(cv_tensor.size[i]);
    }

    // Calculate and copy the strides
    tensor.strides.resize(tensor.shape.rank);
    tensor.strides[tensor.shape.rank - 1] = sizeof(float);
    for (int i = tensor.shape.rank - 2; i >= 0; --i) {
      tensor.strides[i] = tensor.shape.dims[i + 1] * tensor.strides[i + 1];
    }

    // Transfer the CV matrix's data to the ROS2 tensor
    tensor.data.resize(cv_tensor.total() * cv_tensor.elemSize());
    memcpy(tensor.data.data(), cv_tensor.data, tensor.data.size());

    return tensor_list_msg;
  }
};

DnnImageEncoderNode::DnnImageEncoderNode(const rclcpp::NodeOptions options)
: Node("dnn_image_encoder_node", options),
  // Parameters
  network_image_width_(declare_parameter<int>("network_image_width", 224)),
  network_image_height_(declare_parameter<int>("network_image_height", 224)),
  network_image_encoding_(declare_parameter<std::string>("network_image_encoding", "rgb8")),
  tensor_name_(declare_parameter<std::string>("tensor_name", "input")),
  network_normalization_type_(declare_parameter<std::string>(
      "network_normalization_type", "unit_scaling")),
  // Subscriber
  image_sub_(create_subscription<sensor_msgs::msg::Image>(
      "image",
      rclcpp::SensorDataQoS(),
      std::bind(&DnnImageEncoderNode::DnnImageEncoderCallback,
      this, std::placeholders::_1))),
  // Publisher
  tensor_pub_(
    create_publisher<isaac_ros_nvengine_interfaces::msg::TensorList>(
      "encoded_tensor", 1)),
  // Impl initialization
  impl_(std::make_unique<DnnImageEncoderImpl>())
{
  if (g_str_to_image_encoding.find(network_image_encoding_) == g_str_to_image_encoding.end()) {
    throw std::runtime_error(
            "Error: received unsupported network image encoding: " + network_image_encoding_);
  }

  if (g_str_to_normalization_type.find(network_normalization_type_) ==
    g_str_to_normalization_type.end())
  {
    throw std::runtime_error(
            "Error: received unsupported network image normalization type: " +
            network_normalization_type_);
  }

  impl_->Initialize(this);
}

void DnnImageEncoderNode::DnnImageEncoderCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr image_msg)
{
  isaac_ros_nvengine_interfaces::msg::TensorList msg;
  try {
    msg = impl_->OnCallback(image_msg);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  tensor_pub_->publish(msg);
}

DnnImageEncoderNode::~DnnImageEncoderNode() = default;

}  // namespace dnn_inference
}  // namespace isaac_ros

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::dnn_inference::DnnImageEncoderNode)
