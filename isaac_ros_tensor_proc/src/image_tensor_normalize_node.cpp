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

#include "isaac_ros_tensor_proc/image_tensor_normalize_node.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include "cvcuda_conversions/cvcuda_conversions.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "nvcv/TensorDataAccess.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

namespace TensorConv = cvcuda_conversions;

ImageTensorNormalizeNode::ImageTensorNormalizeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("image_tensor_normalize_node", options),
  image_mean_{declare_parameter<std::vector<double>>("mean", {0.5, 0.5, 0.5})},
  image_stddev_{declare_parameter<std::vector<double>>("stddev", {0.5, 0.5, 0.5})},
  input_tensor_name_{declare_parameter<std::string>("input_tensor_name", "tensor")},
  output_tensor_name_{declare_parameter<std::string>("output_tensor_name", "tensor")},
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")}
{
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ImageTensorNormalizeNode");

  std::vector<float> mean_float(image_mean_.begin(), image_mean_.end());
  std::vector<float> stddev_float(image_stddev_.begin(), image_stddev_.end());
  nvcv::TensorShape::ShapeType shape{nvcv::TensorShape::ShapeType{1, 1, 1,
      static_cast<int64_t>(image_mean_.size())}};
  nvcv::TensorShape tensor_shape{shape, nvcv::TENSOR_NHWC};
  mean_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);
  stddev_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);

  auto mean_data = mean_.exportData<nvcv::TensorDataStridedCuda>();
  auto mean_access = nvcv::TensorDataAccessStridedImagePlanar::Create(*mean_data);
  auto stddev_data = stddev_.exportData<nvcv::TensorDataStridedCuda>();
  auto stddev_access = nvcv::TensorDataAccessStridedImagePlanar::Create(*stddev_data);

  cudaError_t err = cudaMemcpy2DAsync(
    mean_access->sampleData(0), mean_access->rowStride(), mean_float.data(),
    mean_float.size() * sizeof(float), mean_float.size() * sizeof(float), 1,
    cudaMemcpyHostToDevice, *cuda_stream_);
  CHECK_CUDA_ERROR(err, "[ImageTensorNormalizeNode] cudaMemcpy2DAsync for mean failed");
  err = cudaMemcpy2DAsync(
    stddev_access->sampleData(0), stddev_access->rowStride(), stddev_float.data(),
    stddev_float.size() * sizeof(float), stddev_float.size() * sizeof(float), 1,
    cudaMemcpyHostToDevice, *cuda_stream_);
  CHECK_CUDA_ERROR(err, "[ImageTensorNormalizeNode] cudaMemcpy2DAsync for stddev failed");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(*cuda_stream_),
    "[ImageTensorNormalizeNode] cudaStreamSynchronize failed");

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  tensor_list_sub_ = create_subscription<TensorList>(
    "tensor", input_qos_,
    std::bind(&ImageTensorNormalizeNode::ImageTensorNormalizeCallback, this,
      std::placeholders::_1), sub_options);
  tensor_list_pub_ = create_publisher<TensorList>(
    "normalized_tensor", output_qos_, pub_options);
  RCLCPP_INFO(get_logger(), "[ImageTensorNormalizeNode] Setup complete");
}

ImageTensorNormalizeNode::~ImageTensorNormalizeNode() {}

void ImageTensorNormalizeNode::ImageTensorNormalizeCallback(
  const TensorList::ConstSharedPtr tensor_msg)
{
  const TensorConv::Tensor * input_tensor =
    TensorConv::find_tensor_by_name(*tensor_msg, input_tensor_name_);
  if (!input_tensor) {
    RCLCPP_ERROR(get_logger(), "[ImageTensorNormalizeNode] Input tensor %s not found",
      input_tensor_name_.c_str());
    return;
  }

  // The CV-CUDA normalize op runs on input and output tensors wrapped with the
  // same layout so the per-channel mean/stddev broadcast lands on the channel
  // axis. For a 3D HWC input we run the op as HWC and then prepend a batch
  // dimension to the published message (NHWC) for downstream consumers.
  nvcv::TensorLayout layout;
  bool prepend_batch_dim = false;
  if (input_tensor->shape.size() == 4) {
    layout = nvcv::TENSOR_NCHW;
  } else if (input_tensor->shape.size() == 3) {
    layout = nvcv::TENSOR_HWC;
    prepend_batch_dim = true;
  } else {
    RCLCPP_ERROR(
      get_logger(),
      "[ImageTensorNormalizeNode] Unsupported input tensor shape rank: %zu",
      input_tensor->shape.size());
    throw std::invalid_argument(
            "[ImageTensorNormalizeNode] Unsupported input tensor shape rank: " +
            std::to_string(input_tensor->shape.size()));
  }

  const std::vector<int64_t> output_shape(
    input_tensor->shape.begin(), input_tensor->shape.end());
  constexpr uint8_t kFloatCode = static_cast<uint8_t>(TensorConv::DLDataTypeCode::kFloat);

  TensorConv::Tensor output_tensor;
  {
    auto input_handle = TensorConv::from_input_tensor(
      *input_tensor, *cuda_stream_, layout);
    output_tensor = TensorConv::allocate_tensor(
      output_shape, kFloatCode, 32, input_tensor->dtype_lanes);
    auto output_handle = TensorConv::from_output_tensor(
      output_tensor, *cuda_stream_, layout);
    normalize_op_(
      *cuda_stream_, input_handle, mean_, stddev_, output_handle,
      1.0f, 0.0f, 0.0f, CVCUDA_NORMALIZE_SCALE_IS_STDDEV);
  }

  // Present the HWC result as NHWC (add batch dim) without touching the buffer.
  if (prepend_batch_dim) {
    output_tensor.shape.insert(output_tensor.shape.begin(), 1);
  }

  TensorList output_list;
  output_list.header = tensor_msg->header;
  output_list.names = {output_tensor_name_};
  output_list.tensors = {std::move(output_tensor)};
  tensor_list_pub_->publish(std::move(output_list));
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode)
