// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_tensor_proc/image_to_tensor_node.hpp"

#include <climits>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_handle.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "std_msgs/msg/header.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

ImageToTensorNode::ImageToTensorNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("image_to_tensor_node", options),
  scale_{declare_parameter<bool>("scale", true)},
  tensor_name_{declare_parameter<std::string>("tensor_name", "tensor")},
  memory_pool_block_size_{declare_parameter<int64_t>("memory_pool_block_size",
    1920 * 1200 * 4 * sizeof(float))},
  memory_pool_num_blocks_{declare_parameter<int64_t>("memory_pool_num_blocks", 40)},
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")}
{
  RCLCPP_DEBUG(get_logger(), "[ImageToTensorNode] Constructor");

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ImageToTensorNode");

  // Create subscribers and publishers
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  image_sub_ = create_subscription<nvidia::isaac_ros::nitros::NitrosImage>(
    "image", input_qos_,
    std::bind(&ImageToTensorNode::ImageToTensorCallback, this, std::placeholders::_1),
    sub_options);
  tensor_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "tensor", output_qos_, pub_options);

  RCLCPP_DEBUG(get_logger(), "[ImageToTensorNode] Setup complete");
}

void ImageToTensorNode::ImageToTensorCallback(
  const nvidia::isaac_ros::nitros::NitrosImage::SharedPtr img_msg)
{
  int32_t num_channels{sensor_msgs::image_encodings::numChannels(img_msg->encoding)};
  int32_t bytes_per_channel = sensor_msgs::image_encodings::bitDepth(img_msg->encoding) / CHAR_BIT;
  const cvcuda_utils::NVCVImageFormat input_format = cvcuda_utils::ToNVCVFormat(img_msg->encoding);
  auto input_handle = cvcuda_utils::WrapCVCUDATensor(
    *img_msg, img_msg->get_read_handle(*cuda_stream_), input_format.format, num_channels,
    bytes_per_channel);

  // Create output tensor
  nvidia::isaac_ros::nitros::NitrosTensor output_tensor;
  // Assuming non-batched image input in HWC format
  std::vector<uint32_t> dims = {img_msg->height, img_msg->width,
    static_cast<uint32_t>(num_channels)};
  nvidia::isaac_ros::nitros::NitrosTensorShape output_tensor_shape{dims};

  // Lazily size the pool to fit the actual output tensor; the configured
  // memory_pool_block_size_ default is not guaranteed to cover all shapes.
  if (!pool_.initialized()) {
    const size_t required_size =
      static_cast<size_t>(img_msg->height) *
      static_cast<size_t>(img_msg->width) *
      static_cast<size_t>(num_channels) * sizeof(float);
    const int64_t actual_block_size = std::max(
      memory_pool_block_size_, static_cast<int64_t>(required_size));
    CHECK_CUDA_ERROR(pool_.create(
      static_cast<size_t>(actual_block_size),
      static_cast<size_t>(memory_pool_num_blocks_),
      nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device),
      "[ImageToTensorNode] Failed to create memory pool");
  }

  auto output_write_handle = output_tensor.from_pool(
    tensor_name_, pool_, output_tensor_shape,
    nvidia::isaac_ros::nitros::NitrosDataType::kFloat32, *cuda_stream_);

  // Create NVCV shape for WrapCVCUDATensor
  nvcv::TensorShape::ShapeType output_nvcv_shape{img_msg->height, img_msg->width, num_channels};
  nvcv::DataType output_data_type = nvcv::TYPE_F32;
  nvcv::TensorLayout output_tensor_layout = nvcv::TENSOR_HWC;
  auto output_handle = cvcuda_utils::WrapCVCUDATensor(
    output_tensor, std::move(output_write_handle), output_nvcv_shape,
    output_data_type, output_tensor_layout);

  const float scale_factor = scale_ ? 1.0f / 255.0f : 1.0f;
  convert_op_(*cuda_stream_, input_handle.get_tensor(), output_handle.get_tensor(), scale_factor,
    0.0f);

  std_msgs::msg::Header header;
  header.stamp.sec = img_msg->get_timestamp_sec();
  header.stamp.nanosec = img_msg->get_timestamp_nsec();
  header.frame_id = img_msg->get_frame_id();
  nvidia::isaac_ros::nitros::NitrosTensorList tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(header)
    .AddTensor(output_tensor)
    .Build();
  tensor_pub_->publish(tensor_list);
}

ImageToTensorNode::~ImageToTensorNode() {}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::ImageToTensorNode)
