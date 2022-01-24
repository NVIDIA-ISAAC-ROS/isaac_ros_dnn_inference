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
#include <vector>

#include "cv_bridge/cv_bridge.h"
#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"

namespace
{
enum NormalizationTypes
{
  kNone,
  kUnitScaling,
  kPositiveNegative,
  kImageNormalization
};

enum TensorLayouts
{
  NCHW,
  NHWC
};

const std::unordered_map<std::string, int32_t> g_str_to_normalization_type({
    {"none", NormalizationTypes::kNone},
    {"unit_scaling", NormalizationTypes::kUnitScaling},
    {"positive_negative", NormalizationTypes::kPositiveNegative},
    {"image_normalization", NormalizationTypes::kImageNormalization}}
);

const std::unordered_map<std::string, int32_t> g_str_to_tensor_layout({
    {"nchw", TensorLayouts::NCHW},
    {"nhwc", TensorLayouts::NHWC}}
);

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
  std::string tensor_layout_;
  bool maintain_aspect_ratio_;
  bool center_crop_;
  std::vector<double> image_mean_;
  std::vector<double> image_stddev_;

  void Initialize(DnnImageEncoderNode * encoder_node)
  {
    node = encoder_node;
    image_width_ = node->network_image_width_;
    image_height_ = node->network_image_height_;
    image_encoding_ = node->network_image_encoding_;
    normalization_type_ = node->network_normalization_type_;
    tensor_name_ = node->tensor_name_;
    tensor_layout_ = node->tensor_layout_;
    maintain_aspect_ratio_ = node->maintain_aspect_ratio_;
    center_crop_ = node->center_crop_;
    image_mean_ = node->image_mean_;
    image_stddev_ = node->image_stddev_;
  }

  isaac_ros_nvengine_interfaces::msg::TensorList OnCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg)
  {
    // Cv bridge conversion and convert the color space
    cv_bridge::CvImagePtr image_ptr;
    image_ptr = cv_bridge::toCvCopy(image_msg, g_str_to_image_encoding.at(image_encoding_));

    // Resize the image to the user specified dimensions
    cv::Mat image_resized;

    if (maintain_aspect_ratio_) {
      const double width_ratio = static_cast<double>(image_msg->width) /
        static_cast<double>(image_width_);
      const double height_ratio = static_cast<double>(image_msg->height) /
        static_cast<double>(image_height_);
      cv::Size size;
      if (height_ratio < width_ratio) {  // Cropping width
        const double target_ratio = static_cast<double>(image_width_) /
          static_cast<double>(image_height_);
        const double crop_height = image_msg->height;
        // Make sure the amount cropped is less than or equal to the current width of image
        const bool cropped_less = target_ratio * image_msg->height < image_msg->width;
        const double crop_width =
          (cropped_less) ? target_ratio * image_msg->height : image_msg->width;
        cv::Rect cropped_area(
          (center_crop_) ? (static_cast<double>(image_msg->width) - crop_width) / 2.0 : 0,
          0, crop_width, crop_height);
        image_ptr->image = image_ptr->image(cropped_area);
      } else {  // Cropping height
        const double target_ratio = static_cast<double>(image_height_) /
          static_cast<double>(image_width_);
        const double crop_width = image_msg->width;
        // Make sure the amount cropped is less than or equal to the current height of image
        const bool cropped_less = target_ratio * image_msg->width < image_msg->height;
        const double crop_height =
          (cropped_less) ? target_ratio * image_msg->width : image_msg->height;
        cv::Rect cropped_area(0,
          (center_crop_) ? (static_cast<double>(image_msg->height) - crop_height) / 2.0 : 0,
          crop_width, crop_height);
        image_ptr->image = image_ptr->image(cropped_area);
      }
    }
    cv::resize(image_ptr->image, image_resized, cv::Size(image_width_, image_height_));

    // Normalize tensor depending on normalization type required
    switch (g_str_to_normalization_type.at(normalization_type_)) {
      case NormalizationTypes::kUnitScaling:
        image_resized.convertTo(image_resized, CV_32F, 1.0f / 255.0f);
        break;
      case NormalizationTypes::kPositiveNegative:
        image_resized.convertTo(image_resized, CV_32F, 2.0f / 255.0f, -1.0f);
        break;
      case NormalizationTypes::kImageNormalization:
        image_resized.convertTo(image_resized, CV_32F);
        image_resized.forEach<cv::Vec3f>(
          [this](cv::Vec3f & pixel, const int *) -> void
          {
            pixel[0] = (pixel[0] / 255.0f - image_mean_[0]) / image_stddev_[0];
            pixel[1] = (pixel[1] / 255.0f - image_mean_[1]) / image_stddev_[1];
            pixel[2] = (pixel[2] / 255.0f - image_mean_[2]) / image_stddev_[2];
          });
        break;
      default:
        image_resized.convertTo(image_resized, CV_32F);
    }

    // Convert tensor layout to the tensor_layout specified by the parameters
    cv::Mat cv_tensor;
    switch (g_str_to_tensor_layout.at(tensor_layout_)) {
      case TensorLayouts::NHWC:
      {
        int sz[] = {1, image_resized.rows, image_resized.cols, image_resized.channels()};
        cv_tensor = cv::Mat(4, sz, CV_32F, image_resized.data);
        break;
      }
      // Default to NCHW
      default:
        cv_tensor = cv::dnn::blobFromImage(image_resized);
    }

    

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
  maintain_aspect_ratio_(declare_parameter<bool>("maintain_aspect_ratio", false)),
  center_crop_(declare_parameter<bool>("center_crop", false)),
  image_mean_(declare_parameter<std::vector<double>>("image_mean", {0.5, 0.5, 0.5})),
  image_stddev_(declare_parameter<std::vector<double>>("image_stddev", {0.5, 0.5, 0.5})),
  tensor_name_(declare_parameter<std::string>("tensor_name", "input")),
  tensor_layout_(declare_parameter<std::string>("tensor_layout", "nchw")),
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
  if (g_str_to_tensor_layout.find(tensor_layout_) == g_str_to_tensor_layout.end()) {
    throw std::runtime_error(
            "Error: received unsupported tensor layout: " + tensor_layout_);
  }

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

  if (network_normalization_type_ == "image_normalization" &&
    (image_mean_.size() != 3 || image_stddev_.size() != 3))
  {
    throw std::runtime_error(
            "Error: if normalization type is set to Image Normalization, vectors image_mean "
            "and image_stddev must have exactly 3 elements");
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
