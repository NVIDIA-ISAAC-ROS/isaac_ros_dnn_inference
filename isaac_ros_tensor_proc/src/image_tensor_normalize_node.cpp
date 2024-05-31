// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_shape.hpp"
#include "nvcv/TensorDataAccess.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

namespace
{

nvcv::TensorLayout StrToTensorLayout(const std::string & str)
{
  // NOTE: cvcuda seems to not work with other formats
  static const std::unordered_map<std::string, nvcv::TensorLayout> str_to_tensor_layout = {
    {"HWC", nvcv::TENSOR_HWC},
  };
  return str_to_tensor_layout.at(str);
}

inline void CheckCudaErrors(cudaError_t code, const char * file, const int line)
{
  if (code != cudaSuccess) {
    const std::string message = "CUDA error returned at " + std::string(file) + ":" +
      std::to_string(line) + ", Error code: " + std::to_string(code) +
      " (" + std::string(cudaGetErrorString(code)) + ")";
    throw std::runtime_error(message);
  }
}

nvcv::TensorShape::ShapeType GetShape(
  const ::nvidia::isaac_ros::nitros::NitrosTensorShape & shape,
  const nvcv::TensorLayout & tensor_layout)
{
  if (tensor_layout == nvcv::TENSOR_HWC || tensor_layout == nvcv::TENSOR_CHW) {
    return {shape.shape().dimension(0), shape.shape().dimension(1), shape.shape().dimension(2)};
  } else if (tensor_layout == nvcv::TENSOR_NHWC || tensor_layout == nvcv::TENSOR_NCHW) {
    return {
      shape.shape().dimension(0), shape.shape().dimension(1), shape.shape().dimension(2),
      shape.shape().dimension(3)};
  }
  throw std::invalid_argument("Error: received unexpected tensor format!");
}


void ComputeBufferStrides(
  const ::nvidia::isaac_ros::nitros::NitrosTensorShape & shape,
  const nvcv::TensorLayout & tensor_layout,
  const size_t bytes_per_element,
  nvcv::TensorDataStridedCuda::Buffer & buffer)
{
  // Manually compute strides, we should get this from CUDA with NITROS later
  if (tensor_layout == nvcv::TENSOR_HWC || tensor_layout == nvcv::TENSOR_CHW) {
    buffer.strides[2] = bytes_per_element;
    buffer.strides[1] = shape.shape().dimension(2) * buffer.strides[2];
    buffer.strides[0] = shape.shape().dimension(1) * buffer.strides[1];
  } else if (tensor_layout == nvcv::TENSOR_NHWC || tensor_layout == nvcv::TENSOR_NCHW) {
    buffer.strides[3] = bytes_per_element;
    buffer.strides[2] = shape.shape().dimension(3) * buffer.strides[3];
    buffer.strides[1] = shape.shape().dimension(2) * buffer.strides[2];
    buffer.strides[0] = shape.shape().dimension(1) * buffer.strides[1];
  }
}

void ToCVCudaTensor(
  const ::nvidia::isaac_ros::nitros::NitrosTensorListView::NitrosTensorView & tensor,
  const nvcv::TensorLayout & tensor_layout, nvcv::Tensor & cv_cuda_tensor)
{
  nvcv::TensorDataStridedCuda::Buffer buffer;
  ComputeBufferStrides(tensor.GetShape(), tensor_layout, tensor.GetBytesPerElement(), buffer);
  buffer.basePtr = const_cast<NVCVByte *>(reinterpret_cast<const NVCVByte *>(tensor.GetBuffer()));
  nvcv::TensorShape::ShapeType shape{GetShape(tensor.GetShape(), tensor_layout)};
  nvcv::TensorShape tensor_shape{shape, tensor_layout};

  auto tensor_type = [&]() {
      switch (tensor.GetElementType()) {
        case nvidia::isaac_ros::nitros::PrimitiveType::kUnsigned8:
          return nvcv::TYPE_U8;
        case nvidia::isaac_ros::nitros::PrimitiveType::kFloat32:
          return nvcv::TYPE_F32;
        default:
          throw std::invalid_argument("Received unexpected type!");
      }
    }();
  nvcv::TensorDataStridedCuda data{tensor_shape, tensor_type, buffer};
  cv_cuda_tensor = nvcv::TensorWrapData(data);
}

void ToCVCudaTensor(
  float * tensor, const ::nvidia::isaac_ros::nitros::NitrosTensorShape & shape,
  const nvcv::TensorLayout & tensor_layout, nvcv::Tensor & cv_cuda_tensor)
{
  nvcv::TensorDataStridedCuda::Buffer buffer;
  ComputeBufferStrides(shape, tensor_layout, sizeof(float), buffer);
  buffer.basePtr = const_cast<NVCVByte *>(reinterpret_cast<const NVCVByte *>(tensor));
  nvcv::TensorShape::ShapeType cv_shape{GetShape(shape, tensor_layout)};
  nvcv::TensorShape tensor_shape{cv_shape, tensor_layout};
  nvcv::TensorDataStridedCuda data{tensor_shape, nvcv::TYPE_F32, buffer};
  cv_cuda_tensor = nvcv::TensorWrapData(data);
}

}  // namespace

ImageTensorNormalizeNode::ImageTensorNormalizeNode(const rclcpp::NodeOptions options)
: rclcpp::Node("image_tensor_normalize_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  nitros_tensor_sub_{std::make_shared<::nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        ::nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this, "tensor",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nhwc_rgb_f32_t::supported_type_name,
      std::bind(&ImageTensorNormalizeNode::ImageTensorNormalizeCallback, this,
      std::placeholders::_1), nvidia::isaac_ros::nitros::NitrosStatisticsConfig{}, input_qos_)},
  nitros_tensor_pub_{std::make_shared<
      nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosTensorList>>(
      this, "normalized_tensor",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nhwc_rgb_f32_t::supported_type_name,
      nvidia::isaac_ros::nitros::NitrosStatisticsConfig{}, output_qos_)},
  mean_param_{declare_parameter<std::vector<double>>("mean", {0.5, 0.5, 0.5})},
  stddev_param_{declare_parameter<std::vector<double>>("stddev", {0.5, 0.5, 0.5})},
  input_tensor_name_{declare_parameter<std::string>("input_tensor_name", "tensor")},
  output_tensor_name_{declare_parameter<std::string>("output_tensor_name", "tensor")},
  tensor_layout_{StrToTensorLayout("HWC")}
{
  CheckCudaErrors(cudaStreamCreate(&stream_), __FILE__, __LINE__);
  std::vector<float> mean_float(mean_param_.begin(), mean_param_.end());
  std::vector<float> stddev_float(stddev_param_.begin(), stddev_param_.end());
  nvcv::TensorShape::ShapeType shape{nvcv::TensorShape::ShapeType{1, 1, 1,
      static_cast<int64_t>(mean_param_.size())}};
  nvcv::TensorShape tensor_shape{shape, nvcv::TENSOR_NHWC};

  mean_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);
  stddev_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);

  auto mean_data = mean_.exportData<nvcv::TensorDataStridedCuda>();
  auto mean_access = nvcv::TensorDataAccessStridedImagePlanar::Create(*mean_data);
  auto stddev_data = stddev_.exportData<nvcv::TensorDataStridedCuda>();
  auto stddev_access = nvcv::TensorDataAccessStridedImagePlanar::Create(*stddev_data);

  CheckCudaErrors(
    cudaMemcpy2D(
      mean_access->sampleData(0), mean_access->rowStride(), mean_float.data(),
      mean_float.size() * sizeof(float), mean_float.size() * sizeof(float), 1,
      cudaMemcpyHostToDevice),
    __FILE__, __LINE__);
  CheckCudaErrors(
    cudaMemcpy2D(
      stddev_access->sampleData(0), stddev_access->rowStride(), stddev_float.data(),
      stddev_float.size() * sizeof(float), stddev_float.size() * sizeof(float), 1,
      cudaMemcpyHostToDevice),
    __FILE__, __LINE__);
}

void ImageTensorNormalizeNode::ImageTensorNormalizeCallback(
  const ::nvidia::isaac_ros::nitros::NitrosTensorListView & tensor_msg)
{
  const auto tensor = tensor_msg.GetNamedTensor(input_tensor_name_);
  nvcv::Tensor input_tensor;
  ToCVCudaTensor(tensor, tensor_layout_, input_tensor);

  float * raw_output_buffer{nullptr};
  const size_t output_buffer_size{tensor.GetElementCount() * sizeof(float)};
  CheckCudaErrors(
    cudaMallocAsync(&raw_output_buffer, output_buffer_size, stream_), __FILE__, __LINE__);
  nvcv::Tensor output_tensor;
  ToCVCudaTensor(raw_output_buffer, tensor.GetShape(), tensor_layout_, output_tensor);

  norm_op_(
    stream_, input_tensor, mean_, stddev_, output_tensor, 1.0f, 0.0f, 0.0f,
    CVCUDA_NORMALIZE_SCALE_IS_STDDEV);

  CheckCudaErrors(cudaStreamSynchronize(stream_), __FILE__, __LINE__);

  std_msgs::msg::Header header;
  header.frame_id = tensor_msg.GetFrameId();
  header.stamp.sec = tensor_msg.GetTimestampSeconds();
  header.stamp.nanosec = tensor_msg.GetTimestampNanoseconds();

  nvidia::isaac_ros::nitros::NitrosTensorList tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(header)
    .AddTensor(
    output_tensor_name_, (nvidia::isaac_ros::nitros::NitrosTensorBuilder()
    .WithShape(tensor.GetShape())
    .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kFloat32)
    .WithData(raw_output_buffer)
    .Build()))
    .Build();

  nitros_tensor_pub_->publish(tensor_list);
}

ImageTensorNormalizeNode::~ImageTensorNormalizeNode()
{
  CheckCudaErrors(cudaStreamDestroy(stream_), __FILE__, __LINE__);
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode)
