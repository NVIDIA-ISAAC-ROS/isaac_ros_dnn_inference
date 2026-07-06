// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuda_runtime.h"  // NOLINT - include .h without directory
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/exact_time.h"
#include "message_filters/synchronizer.h"

#include "cvcuda/OpConvertTo.hpp"
#include "cvcuda/OpCopyMakeBorder.hpp"
#include "cvcuda/OpCustomCrop.hpp"
#include "cvcuda/OpNormalize.hpp"
#include "cvcuda/OpReformat.hpp"
#include "cvcuda/OpResize.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_message_filters_subscriber.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "nvcv/Tensor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{
class DnnImageEncoderNode : public rclcpp::Node
{
public:
  explicit DnnImageEncoderNode(const rclcpp::NodeOptions & options);

  ~DnnImageEncoderNode();

private:
  void CalculateResizeAndCropParams();
  float GetResizeScalar();

  void ImageSubCallback(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);

  // Parameters
  std::string tensor_name_;
  const int32_t input_image_width_;
  const int32_t input_image_height_;
  const int32_t network_image_width_;
  const int32_t network_image_height_;
  const std::string input_encoding_;
  const bool enable_padding_;
  const std::vector<double> image_mean_;
  const std::vector<double> image_stddev_;
  const uint16_t batch_size_;
  int64_t memory_pool_block_size_;
  int64_t memory_pool_num_blocks_;
  int64_t input_queue_size_;
  int64_t output_queue_size_;

  // Subscriber and Publisher
  ::message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImage> image_sub_;
  ::message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosTensorList>::SharedPtr tensor_list_pub_;

  using ExactPolicy = ::message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage,
    sensor_msgs::msg::CameraInfo>;
  ::message_filters::Synchronizer<ExactPolicy> exact_sync_;

  // CUDA resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;

  // Internal parameters
  nvcv::Tensor stddev_;
  nvcv::Tensor mean_;
  nvcv::Tensor resized_tensor_;
  nvcv::Tensor padded_tensor_;
  nvcv::Tensor float_tensor_;
  nvcv::Tensor normalized_tensor_;
  nvcv::Tensor planar_tensor_;
  int32_t resize_out_img_width_;
  int32_t resize_out_img_height_;

  // Operators
  cvcuda::Resize resize_op_;
  cvcuda::CopyMakeBorder copy_make_border_op_;
  cvcuda::Reformat reformat_op_;
  cvcuda::Normalize norm_op_;
  cvcuda::ConvertTo convert_op_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DNN_IMAGE_ENCODER__DNN_IMAGE_ENCODER_NODE_HPP_
