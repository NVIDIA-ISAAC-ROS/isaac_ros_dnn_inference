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

#include <climits>
#include <stdexcept>
#include <string>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_handle.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "nvcv/TensorDataAccess.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

ImageTensorNormalizeNode::ImageTensorNormalizeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("image_tensor_normalize_node", options),
  image_mean_{declare_parameter<std::vector<double>>("mean", {0.5, 0.5, 0.5})},
  image_stddev_{declare_parameter<std::vector<double>>("stddev", {0.5, 0.5, 0.5})},
  input_tensor_name_{declare_parameter<std::string>("input_tensor_name", "tensor")},
  output_tensor_name_{declare_parameter<std::string>("output_tensor_name", "tensor")},
  memory_pool_block_size_{declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)},
  memory_pool_num_blocks_{declare_parameter<int64_t>("memory_pool_num_blocks", 40)},
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")}
{
  // Create CUDA resources
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ImageTensorNormalizeNode");

  // Create mean and stddev tensors
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

  // Copy mean and stddev to GPU
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

  // Create subscriber and publisher
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  tensor_list_sub_ = create_subscription<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "tensor", input_qos_,
    std::bind(&ImageTensorNormalizeNode::ImageTensorNormalizeCallback, this,
      std::placeholders::_1), sub_options);
  tensor_list_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "normalized_tensor",
    output_qos_, pub_options);
  RCLCPP_INFO(get_logger(), "[ImageTensorNormalizeNode] Setup complete");
}

ImageTensorNormalizeNode::~ImageTensorNormalizeNode() {}

void ImageTensorNormalizeNode::ImageTensorNormalizeCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::SharedPtr tensor_msg)
{
  // Get input tensor
  std::shared_ptr<nvidia::isaac_ros::nitros::NitrosTensor> input_tensor =
    tensor_msg->get_tensor_by_name(input_tensor_name_);
  if (input_tensor == nullptr) {
    RCLCPP_ERROR(get_logger(), "[ImageTensorNormalizeNode] Input tensor %s not found",
      input_tensor_name_.c_str());
    return;
  }
  nvcv::DataType dtype = cvcuda_utils::ToNVCVDataType(input_tensor->data_type());
  std::vector<int32_t> dims;
  nvcv::TensorShape::ShapeType input_shape;
  nvcv::TensorLayout input_layout;
  for (size_t i = 0; i < input_tensor->shape().rank(); i++) {
    dims.push_back(input_tensor->shape().dims()[i]);
  }
  if (input_tensor->shape().rank() == 4) {
    input_shape = {dims[0], dims[1], dims[2], dims[3]};
    input_layout = nvcv::TENSOR_NCHW;
  } else if (input_tensor->shape().rank() == 3) {
    input_shape = {dims[0], dims[1], dims[2]};
    input_layout = nvcv::TENSOR_HWC;
  } else {
    RCLCPP_ERROR(get_logger(),
      "[ImageTensorNormalizeNode] Unsupported input tensor shape rank: %d",
      input_tensor->shape().rank());
    throw std::invalid_argument(
      "[ImageTensorNormalizeNode] Unsupported input tensor shape rank: " +
      std::to_string(input_tensor->shape().rank()));
  }

  // Wrap input tensor
  auto input_handle = cvcuda_utils::WrapCVCUDATensor(
    *input_tensor, input_tensor->get_read_handle(*cuda_stream_), input_shape, dtype,
    input_layout);

  // Create output tensor
  nvidia::isaac_ros::nitros::NitrosTensor output_tensor;
  nvidia::isaac_ros::nitros::NitrosTensorShape output_tensor_shape;
  if (input_tensor->shape().rank() == 4) {
    output_tensor_shape = nvidia::isaac_ros::nitros::NitrosTensorShape{
      dims[0], dims[1], dims[2], dims[3]};
  } else {
    // Convert HWC to NHWC by adding batch dimension
    output_tensor_shape = nvidia::isaac_ros::nitros::NitrosTensorShape{
      1, dims[0], dims[1], dims[2]};
  }

  const size_t required_size = static_cast<size_t>(dims[0]) * dims[1] * dims[2] * sizeof(float);
  if (!pool_.initialized()) {
    const int64_t actual_block_size = std::max(
      memory_pool_block_size_, static_cast<int64_t>(required_size));
    CHECK_CUDA_ERROR(pool_.create(
      static_cast<size_t>(actual_block_size),
      static_cast<size_t>(memory_pool_num_blocks_),
      nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device),
      "[ImageTensorNormalizeNode] Failed to create CUDA memory pool");
  }
  auto output_write_handle = output_tensor.from_pool(
    output_tensor_name_, pool_, output_tensor_shape,
    nvidia::isaac_ros::nitros::NitrosDataType::kFloat32, *cuda_stream_);
  auto output_handle = cvcuda_utils::WrapCVCUDATensor(
    output_tensor, std::move(output_write_handle), input_shape, nvcv::TYPE_F32, input_layout);

  // Normalize input tensor
  normalize_op_(
    *cuda_stream_, input_handle.get_tensor(), mean_, stddev_, output_handle.get_tensor(),
    1.0f, 0.0f, 0.0f, CVCUDA_NORMALIZE_SCALE_IS_STDDEV);

  nvidia::isaac_ros::nitros::NitrosTensorList tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(tensor_msg->get_header())
    .AddTensor(output_tensor)
    .Build();
  tensor_list_pub_->publish(tensor_list);
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode)
