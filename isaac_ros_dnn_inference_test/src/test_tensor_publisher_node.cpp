/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_dnn_inference_test/test_tensor_publisher_node.hpp"

#include <filesystem>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "isaac_ros_nvengine_interfaces/msg/tensor_shape.hpp"


namespace isaac_ros
{
namespace dnn_inference
{

using namespace std::chrono_literals;

TestTensorPublisherNode::TestTensorPublisherNode(
  const rclcpp::NodeOptions & options,
  std::vector<uint> dimensions)
: Node("test_tensor_publisher", options),
  tensor_name_(declare_parameter<std::string>("tensor_name", "input")),
  data_type_(declare_parameter<int>("data_type", 9)),
  // 150528 derived from default dimensions (1x2x224x224)
  length_(declare_parameter<int>("length", 150528)),
  rank_(declare_parameter<int>("rank", 4)),
  pub_channel_(declare_parameter<std::string>("pub_channel", "tensor_pub"))
{
  pub_ = this->create_publisher<isaac_ros_nvengine_interfaces::msg::TensorList>(pub_channel_, 10);
  timer_ = this->create_wall_timer(
    1000ms, std::bind(&TestTensorPublisherNode::timer_callback, this));

  dimensions_ = dimensions;
}

TestTensorPublisherNode::TestTensorPublisherNode(const rclcpp::NodeOptions & options)
: TestTensorPublisherNode(options, {1, 3, 224, 224}) {}

void TestTensorPublisherNode::timer_callback()
{
  auto tensor = isaac_ros_nvengine_interfaces::msg::Tensor();
  tensor.name = tensor_name_;
  tensor.data_type = data_type_;
  tensor.strides = {};  // Not used in GXF so leave empty

  switch (data_type_) {
    case 9:
      setTensorData<float>(tensor, length_);
      break;
  }

  auto tensor_shape = isaac_ros_nvengine_interfaces::msg::TensorShape();
  tensor_shape.rank = rank_;
  tensor_shape.dims = dimensions_;
  tensor.shape = tensor_shape;

  auto tensor_list = isaac_ros_nvengine_interfaces::msg::TensorList();
  tensor_list.tensors = {tensor};
  pub_->publish(tensor_list);
}

TestTensorPublisherNode::~TestTensorPublisherNode() {}

}  // namespace dnn_inference
}  // namespace isaac_ros
