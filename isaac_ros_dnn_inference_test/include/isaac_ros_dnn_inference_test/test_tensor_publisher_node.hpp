/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_DNN_INFERENCE_TEST__TEST_TENSOR_PUBLISHER_NODE_HPP_
#define ISAAC_ROS_DNN_INFERENCE_TEST__TEST_TENSOR_PUBLISHER_NODE_HPP_

#include <string>
#include <vector>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nvengine_interfaces/msg/tensor.hpp"
#include "isaac_ros_nvengine_interfaces/msg/tensor_list.hpp"


namespace isaac_ros
{
namespace dnn_inference
{

// Sends test ROS tensors of specified dimension to specified channel
// By default this node will send input tensors compatible with mobilenetv2-1.0
// Tensors being sent in are of the dimension 1x3x224x224 as specified by mobilenet
class TestTensorPublisherNode : public rclcpp::Node
{
public:
  explicit TestTensorPublisherNode(const rclcpp::NodeOptions &);

  TestTensorPublisherNode(
    const rclcpp::NodeOptions & options,
    std::vector<uint> dimensions);

  ~TestTensorPublisherNode();

  TestTensorPublisherNode(const TestTensorPublisherNode &) = delete;

  TestTensorPublisherNode & operator=(const TestTensorPublisherNode &) = delete;

private:
  // Name of tensor
  const std::string tensor_name_;
  // Enum of type of data (check Tensor.msg for mappings)
  const int data_type_;
  // Length of tensor when flattened to 1D
  const int length_;
  // Rank of tensor
  const int rank_;
  // Dimensions of tensor (eg. {3, 255, 255} for RGB Image of 255x255)
  std::vector<uint> dimensions_;
  // Publisher channel
  std::string pub_channel_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<isaac_ros_nvengine_interfaces::msg::TensorList>::SharedPtr pub_;

  // Sets the tensor data based on size and type (supports only: float32)
  template<typename T>
  void setTensorData(isaac_ros_nvengine_interfaces::msg::Tensor & tensor, int length)
  {
    std::vector<T> test_data(length);
    tensor.data.resize(test_data.size() * sizeof(T));
    memcpy(tensor.data.data(), test_data.data(), test_data.size() * sizeof(T));
  }

  void timer_callback();
};

}  // namespace dnn_inference
}  // namespace isaac_ros

#endif  // ISAAC_ROS_DNN_INFERENCE_TEST__TEST_TENSOR_PUBLISHER_NODE_HPP_
