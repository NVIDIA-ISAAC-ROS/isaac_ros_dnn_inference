// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TENSORRT_CONVERSIONS__TENSORRT_CONVERSIONS_HPP_
#define TENSORRT_CONVERSIONS__TENSORRT_CONVERSIONS_HPP_

// Header-only converter between a ROS 2 tensor_msgs/ExperimentalTensor (whose
// `data` field is a rosidl::Buffer<uint8_t>) and the native device binding a
// TensorRT nvinfer1::IExecutionContext expects: a device pointer plus an
// nvinfer1::Dims shape and nvinfer1::DataType, riding on whichever rosidl::Buffer
// backend is registered at runtime (e.g. cuda_buffer).
//
// Tensor is DLPack-aligned: the element type is a DLPack triple
// {dtype_code, dtype_bits, dtype_lanes} rather than a single ordinal, the shape
// is a flat int64[], strides are in elements (empty == contiguous row-major),
// and `byte_offset` allows zero-copy views into a larger allocation.
//
// This mirrors the cvcuda_conversions design (allocate_* / from_input_* /
// from_output_* / to_*), but for TensorRT. A TensorRT binding does NOT own its
// memory and enqueueV3 is asynchronous, so from_input/output return a
// TensorBinding holder that keeps the buffer Read/WriteHandle alive until the
// recorded completion event fires. The holder must outlive the enqueueV3 call
// (bind its data() with setTensorAddress) and be destroyed only after submit so
// the completion event is recorded after TensorRT actually reads/writes.
//
// It depends only on tensor_msgs, cuda_buffer, and TensorRT, so it is upstreamable
// next to cvcuda_conversions.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <NvInfer.h>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace tensorrt_conversions
{

using Tensor = tensor_msgs::msg::ExperimentalTensor;

// DLPack DLDataTypeCode values, the subset transportable to TensorRT. Codes >= 7
// (the FP8/FP6/FP4 families) have no whole-byte TensorRT binding type here. See
// https://dmlc.github.io/dlpack/latest/ and ExperimentalTensor.msg.
enum class DLDataTypeCode : uint8_t
{
  kInt = 0,
  kUInt = 1,
  kFloat = 2,
  kBFloat = 4,
  kBool = 6,
};

// A DLPack DLDataType: {code, bits, lanes}. `lanes` is the vector lane count
// (1 for a plain scalar); TensorRT bindings have no vectorized element types.
struct DLDataType
{
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

// Byte size of one element for a DLPack dtype: bits * lanes / 8. TensorRT binding
// types are all whole-byte, so bits must be a positive multiple of 8.
inline int bytes_per_element(uint8_t dtype_bits, uint16_t dtype_lanes)
{
  if (dtype_bits == 0 || (dtype_bits % 8) != 0) {
    throw std::invalid_argument(
            "tensorrt_conversions: dtype_bits must be a positive multiple of 8, got " +
            std::to_string(dtype_bits));
  }
  return static_cast<int>(dtype_bits / 8) * static_cast<int>(dtype_lanes);
}

// Map a DLPack dtype {code, bits, lanes} to an nvinfer1::DataType. TensorRT has
// no int16/uint16/uint32/uint64/float64 binding types (those throw), and no
// vectorized lanes (dtype_lanes must be 1).
inline nvinfer1::DataType to_trt_data_type(
  uint8_t dtype_code, uint8_t dtype_bits, uint16_t dtype_lanes)
{
  if (dtype_lanes != 1) {
    throw std::invalid_argument(
            "tensorrt_conversions: only scalar tensors (dtype_lanes == 1) are supported, got "
            "lanes=" + std::to_string(dtype_lanes));
  }
  switch (static_cast<DLDataTypeCode>(dtype_code)) {
    case DLDataTypeCode::kInt:
      switch (dtype_bits) {
        case 8: return nvinfer1::DataType::kINT8;
        case 32: return nvinfer1::DataType::kINT32;
        case 64: return nvinfer1::DataType::kINT64;
      }
      break;
    case DLDataTypeCode::kUInt:
      if (dtype_bits == 8) {return nvinfer1::DataType::kUINT8;}
      break;
    case DLDataTypeCode::kFloat:
      switch (dtype_bits) {
        case 16: return nvinfer1::DataType::kHALF;
        case 32: return nvinfer1::DataType::kFLOAT;
      }
      break;
    case DLDataTypeCode::kBFloat:
      if (dtype_bits == 16) {return nvinfer1::DataType::kBF16;}
      break;
    case DLDataTypeCode::kBool:
      if (dtype_bits == 8) {return nvinfer1::DataType::kBOOL;}
      break;
  }
  throw std::invalid_argument(
          "tensorrt_conversions: DLPack dtype {code=" + std::to_string(dtype_code) +
          ", bits=" + std::to_string(dtype_bits) +
          "} not representable as a TensorRT binding type");
}

// Map an nvinfer1::DataType back to a DLPack dtype {code, bits, lanes=1}.
inline DLDataType from_trt_data_type(nvinfer1::DataType dtype)
{
  constexpr auto kInt = static_cast<uint8_t>(DLDataTypeCode::kInt);
  constexpr auto kUInt = static_cast<uint8_t>(DLDataTypeCode::kUInt);
  constexpr auto kFloat = static_cast<uint8_t>(DLDataTypeCode::kFloat);
  constexpr auto kBFloat = static_cast<uint8_t>(DLDataTypeCode::kBFloat);
  constexpr auto kBool = static_cast<uint8_t>(DLDataTypeCode::kBool);
  switch (dtype) {
    case nvinfer1::DataType::kINT8: return {kInt, 8, 1};
    case nvinfer1::DataType::kINT32: return {kInt, 32, 1};
    case nvinfer1::DataType::kINT64: return {kInt, 64, 1};
    case nvinfer1::DataType::kUINT8: return {kUInt, 8, 1};
    case nvinfer1::DataType::kHALF: return {kFloat, 16, 1};
    case nvinfer1::DataType::kFLOAT: return {kFloat, 32, 1};
    case nvinfer1::DataType::kBF16: return {kBFloat, 16, 1};
    case nvinfer1::DataType::kBOOL: return {kBool, 8, 1};
    default:
      throw std::invalid_argument(
              "tensorrt_conversions: nvinfer1::DataType not representable as a "
              "DLPack dtype: " + std::to_string(static_cast<int32_t>(dtype)));
  }
}

// Build an nvinfer1::Dims from an ExperimentalTensor's int64[] shape.
inline nvinfer1::Dims to_trt_dims(const Tensor & tensor)
{
  if (tensor.shape.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS)) {
    throw std::invalid_argument(
            "tensorrt_conversions: tensor rank " + std::to_string(tensor.shape.size()) +
            " exceeds nvinfer1::Dims::MAX_DIMS");
  }
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int32_t>(tensor.shape.size());
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    dims.d[i] = static_cast<int64_t>(tensor.shape[i]);
  }
  return dims;
}

// Product of dimensions (element count) of an nvinfer1::Dims. TensorRT marks
// dynamic dimensions with -1 (e.g. from getTensorShape() before setInputShape);
// throw rather than wrap the unsigned accumulator on any negative extent.
inline size_t num_elements(const nvinfer1::Dims & dims)
{
  size_t count = 1;
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      throw std::invalid_argument(
              "tensorrt_conversions: dynamic/negative dimension " +
              std::to_string(dims.d[i]) + " at index " + std::to_string(i) +
              "; resolve the shape (e.g. setInputShape) before computing element count");
    }
    count *= static_cast<size_t>(dims.d[i]);
  }
  return count;
}

namespace detail
{

// Cast a byte pointer to void while preserving its const-ness, so the
// pointer a binding exposes stays as read-only or writable as its handle.
inline void * as_void_ptr(uint8_t * ptr) {return ptr;}
inline const void * as_void_ptr(const uint8_t * ptr) {return ptr;}

// Bind a read (input) tensor to an execution context: set its resolved shape,
// then its device address. setTensorAddress takes a non-const void*, but
// TensorRT never writes an input binding, so the const_cast that requires is
// localized here at the API boundary. Throws on TensorRT failure.
//
// Templated on the context type so a unit test can drive it with a recording
// double; a real nvinfer1::IExecutionContext has non-virtual setInputShape/
// setTensorAddress and needs a deserialized engine, so it cannot be faked.
template<typename ContextT>
inline void bind_input(
  ContextT & context, const char * name, const nvinfer1::Dims & dims,
  const void * address)
{
  if (!context.setInputShape(name, dims)) {
    throw std::runtime_error(
            std::string("tensorrt_conversions: setInputShape failed for input '") + name + "'");
  }
  if (!context.setTensorAddress(name, const_cast<void *>(address))) {
    throw std::runtime_error(
            std::string("tensorrt_conversions: setTensorAddress failed for input '") +
            name + "'");
  }
}

// Bind a write (output) tensor's device address; the engine determines the
// output shape, so only the address is set. Throws on TensorRT failure.
template<typename ContextT>
inline void bind_output(ContextT & context, const char * name, void * address)
{
  if (!context.setTensorAddress(name, address)) {
    throw std::runtime_error(
            std::string("tensorrt_conversions: setTensorAddress failed for output '") +
            name + "'");
  }
}

// Enqueue inference on the given CUDA stream for a context whose input/output
// bindings are already set. Throws on TensorRT failure. Templated on the context
// type for the same testability reason as bind_input/bind_output above.
template<typename ContextT>
inline void enqueue(ContextT & context, cudaStream_t stream)
{
  if (!context.enqueueV3(stream)) {
    throw std::runtime_error("tensorrt_conversions: enqueueV3 failed");
  }
}

}  // namespace detail

// Owns the buffer Read/WriteHandle for the lifetime of the TensorRT binding, and
// caches the shape/dtype/size needed to bind it to an execution context.
template<typename HandleT>
class TensorBinding
{
public:
  TensorBinding() = default;

  TensorBinding(
    HandleT handle, std::string name, nvinfer1::Dims dims, nvinfer1::DataType dtype,
    size_t size_bytes, size_t byte_offset = 0)
  : handle_(std::move(handle)), name_(std::move(name)), dims_(dims), dtype_(dtype),
    size_bytes_(size_bytes), byte_offset_(byte_offset) {}

  TensorBinding(const TensorBinding &) = delete;
  TensorBinding & operator=(const TensorBinding &) = delete;
  TensorBinding(TensorBinding &&) = default;
  TensorBinding & operator=(TensorBinding &&) = default;
  ~TensorBinding() = default;

  // Device pointer to bind with IExecutionContext::setTensorAddress, advanced by
  // the tensor's byte_offset (nonzero for a view into a larger allocation). Its
  // const-ness follows the handle: a ReadHandle (input binding) yields a const
  // void*, a WriteHandle (output binding) yields a void*. TensorRT's
  // setTensorAddress takes a non-const void*, so a caller binding an input must
  // const_cast at that call site -- where the API, which never writes input
  // bindings, forces it -- rather than this accessor silently doing so.
  auto data() {return detail::as_void_ptr(handle_.get_ptr() + byte_offset_);}

  const nvinfer1::Dims & dims() const {return dims_;}
  nvinfer1::DataType data_type() const {return dtype_;}
  size_t size_bytes() const {return size_bytes_;}

  // Bind this read binding as the named input of an execution context: sets the
  // (already resolved) input shape and the device address in one call. The bind
  // logic lives in detail::bind_input (unit-tested with a recording context);
  // this forwards the binding's resolved dims and device pointer. Throws on
  // TensorRT failure.
  void as_input(nvinfer1::IExecutionContext & context, const char * name)
  {
    detail::bind_input(context, name, dims_, data());
  }

  // Bind this write binding as the named output of an execution context: sets the
  // device address (the engine determines the output shape). Throws on failure.
  void as_output(nvinfer1::IExecutionContext & context, const char * name)
  {
    detail::bind_output(context, name, data());
  }

private:
  HandleT handle_;
  std::string name_{};
  nvinfer1::Dims dims_{};
  nvinfer1::DataType dtype_{};
  size_t size_bytes_{0};
  size_t byte_offset_{0};
};

// Enqueue inference on `context` on the given CUDA stream. Every input and output
// tensor must already be bound (see TensorBinding::as_input / as_output). The
// call is asynchronous: it returns once the work is queued, not once it
// completes -- observe completion by synchronizing the stream (or waiting on the
// cuda_buffer handles' events, which the WriteHandle records on release). Throws
// on TensorRT failure.
inline void enqueue(nvinfer1::IExecutionContext & context, cudaStream_t stream)
{
  detail::enqueue(context, stream);
}

// Allocate a CUDA-backed ExperimentalTensor sized for (shape, dtype). Fills
// dtype/shape/data with a contiguous row-major layout: strides is left empty
// (the DLPack convention for "contiguous, infer row-major from shape") and
// byte_offset is 0. ExperimentalTensor has no name field; the TensorRT binding
// name is supplied separately by the caller from the engine's I/O tensor names.
inline Tensor allocate_tensor(
  const std::vector<int64_t> & shape, uint8_t dtype_code, uint8_t dtype_bits,
  uint16_t dtype_lanes = 1)
{
  const int element_size = bytes_per_element(dtype_bits, dtype_lanes);

  Tensor tensor;
  tensor.dtype_code = dtype_code;
  tensor.dtype_bits = dtype_bits;
  tensor.dtype_lanes = dtype_lanes;
  tensor.shape.assign(shape.begin(), shape.end());
  tensor.byte_offset = 0;

  const size_t num = num_elements(to_trt_dims(tensor));
  tensor.data = cuda_buffer_backend::allocate_buffer(num * element_size);
  return tensor;
}

// Read binding over tensor.data; shape/dtype/offset derived from the message.
inline TensorBinding<cuda_buffer_backend::ReadHandle> from_input_tensor(
  std::string name, const Tensor & tensor, cudaStream_t stream)
{
  auto handle = cuda_buffer_backend::from_input_buffer(tensor.data, stream);
  const nvinfer1::Dims dims = to_trt_dims(tensor);
  const nvinfer1::DataType dtype = to_trt_data_type(
    tensor.dtype_code, tensor.dtype_bits, tensor.dtype_lanes);
  const size_t size_bytes = num_elements(dims) *
    bytes_per_element(tensor.dtype_bits, tensor.dtype_lanes);
  return TensorBinding<cuda_buffer_backend::ReadHandle>(
    std::move(handle), name, dims, dtype, size_bytes, tensor.byte_offset);
}

// Write binding over tensor.data; shape/dtype/offset derived from the message.
inline TensorBinding<cuda_buffer_backend::WriteHandle> from_output_tensor(
  std::string name, Tensor & tensor, cudaStream_t stream)
{
  auto handle = cuda_buffer_backend::from_output_buffer(tensor.data, stream);
  const nvinfer1::Dims dims = to_trt_dims(tensor);
  const nvinfer1::DataType dtype = to_trt_data_type(
    tensor.dtype_code, tensor.dtype_bits, tensor.dtype_lanes);
  const size_t size_bytes = num_elements(dims) *
    bytes_per_element(tensor.dtype_bits, tensor.dtype_lanes);
  return TensorBinding<cuda_buffer_backend::WriteHandle>(
    std::move(handle), name, dims, dtype, size_bytes, tensor.byte_offset);
}

// Copy a standalone device buffer (e.g. a TensorRT output bound to a pool) into
// a pre-allocated Tensor message's buffer. Unlike from_output_tensor (which is
// zero-copy: TensorRT writes directly into the message), this performs a
// device-to-device copy for producers whose result is not already backed by the
// message. `src_size_bytes` must not exceed the message's allocation.
inline void to_tensor(
  Tensor & tensor, const void * src_device_ptr, size_t src_size_bytes, cudaStream_t stream)
{
  auto handle = cuda_buffer_backend::from_output_buffer(tensor.data, stream);
  cudaError_t err = cudaMemcpyAsync(
    handle.get_ptr(), src_device_ptr, src_size_bytes,
    cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
            std::string("tensorrt_conversions::to_tensor: cudaMemcpyAsync failed: ") +
            cudaGetErrorString(err));
  }
}

// Allocate a CUDA-backed Tensor for (shape, dtype) and copy a device
// buffer into it.
inline Tensor to_tensor(
  const std::vector<int64_t> & shape, uint8_t dtype_code, uint8_t dtype_bits,
  const void * src_device_ptr, cudaStream_t stream, uint16_t dtype_lanes = 1)
{
  Tensor tensor = allocate_tensor(shape, dtype_code, dtype_bits, dtype_lanes);
  const size_t size_bytes = num_elements(to_trt_dims(tensor)) *
    bytes_per_element(dtype_bits, dtype_lanes);
  to_tensor(tensor, src_device_ptr, size_bytes, stream);
  return tensor;
}

}  // namespace tensorrt_conversions
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // TENSORRT_CONVERSIONS__TENSORRT_CONVERSIONS_HPP_
