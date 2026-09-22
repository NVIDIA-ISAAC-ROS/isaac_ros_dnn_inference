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

#ifndef TRITON_CONVERSIONS__TRITON_CONVERSIONS_HPP_
#define TRITON_CONVERSIONS__TRITON_CONVERSIONS_HPP_

// Header-only converter between a ROS 2 tensor_msgs/ExperimentalTensor (whose
// `data` field is a rosidl::Buffer<uint8_t>) and the native inputs the Triton
// in-process C API (tritonserver.h) expects: a device pointer plus an int64
// shape and a TRITONSERVER_DataType, riding on whichever rosidl::Buffer backend
// is registered at runtime (e.g. cuda_buffer).
//
// ExperimentalTensor is DLPack-aligned: the element type is a DLPack triple
// {dtype_code, dtype_bits, dtype_lanes} rather than a single ordinal, the shape
// is a flat int64[], strides are in elements (empty == contiguous row-major),
// and `byte_offset` allows zero-copy views into a larger allocation.
//
// This mirrors the cvcuda_conversions design (allocate_* / from_input_* /
// from_output_* / to_*), but for Triton. The device buffer appended via
// TRITONSERVER_InferenceRequestAppendInputData is not owned by Triton and
// TRITONSERVER_ServerInferAsync is asynchronous, so from_input/output return a
// TritonTensor holder that keeps the buffer Read/WriteHandle alive until the
// recorded completion event fires. The holder must outlive the inference call
// and be destroyed only after it completes so the completion event is recorded
// after Triton actually reads/writes.
//
// It depends only on tensor_msgs, cuda_buffer, and Triton, so it is upstreamable
// next to cvcuda_conversions.

#include <cuda_runtime.h>

#include <triton/core/tritonserver.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace triton_conversions
{

using Tensor = tensor_msgs::msg::ExperimentalTensor;
using TensorData = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<Tensor>().data)>>;

// DLPack DLDataTypeCode values, the subset transportable to Triton. Codes >= 7
// (the FP8/FP6/FP4 families) have no Triton binding type. See
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
// (1 for a plain scalar); Triton has no vectorized element types.
struct DLDataType
{
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

// Byte size of one element for a DLPack dtype: bits * lanes / 8. Triton binding
// types are all whole-byte, so bits must be a positive multiple of 8.
inline int bytes_per_element(uint8_t dtype_bits, uint16_t dtype_lanes)
{
  if (dtype_bits == 0 || (dtype_bits % 8) != 0) {
    throw std::invalid_argument(
            "triton_conversions: dtype_bits must be a positive multiple of 8, got " +
            std::to_string(dtype_bits));
  }
  return static_cast<int>(dtype_bits / 8) * static_cast<int>(dtype_lanes);
}

// Map a DLPack dtype {code, bits, lanes} to a TRITONSERVER_DataType. Triton has
// no vectorized lanes, so dtype_lanes must be 1.
inline TRITONSERVER_DataType to_triton_data_type(
  uint8_t dtype_code, uint8_t dtype_bits, uint16_t dtype_lanes)
{
  if (dtype_lanes != 1) {
    throw std::invalid_argument(
            "triton_conversions: only scalar tensors (dtype_lanes == 1) are supported, got "
            "lanes=" + std::to_string(dtype_lanes));
  }
  switch (static_cast<DLDataTypeCode>(dtype_code)) {
    case DLDataTypeCode::kInt:
      switch (dtype_bits) {
        case 8: return TRITONSERVER_TYPE_INT8;
        case 16: return TRITONSERVER_TYPE_INT16;
        case 32: return TRITONSERVER_TYPE_INT32;
        case 64: return TRITONSERVER_TYPE_INT64;
      }
      break;
    case DLDataTypeCode::kUInt:
      switch (dtype_bits) {
        case 8: return TRITONSERVER_TYPE_UINT8;
        case 16: return TRITONSERVER_TYPE_UINT16;
        case 32: return TRITONSERVER_TYPE_UINT32;
        case 64: return TRITONSERVER_TYPE_UINT64;
      }
      break;
    case DLDataTypeCode::kFloat:
      switch (dtype_bits) {
        case 16: return TRITONSERVER_TYPE_FP16;
        case 32: return TRITONSERVER_TYPE_FP32;
        case 64: return TRITONSERVER_TYPE_FP64;
      }
      break;
    case DLDataTypeCode::kBFloat:
      if (dtype_bits == 16) {return TRITONSERVER_TYPE_BF16;}
      break;
    case DLDataTypeCode::kBool:
      if (dtype_bits == 8) {return TRITONSERVER_TYPE_BOOL;}
      break;
  }
  throw std::invalid_argument(
          "triton_conversions: DLPack dtype {code=" + std::to_string(dtype_code) +
          ", bits=" + std::to_string(dtype_bits) +
          "} not representable as a TRITONSERVER_DataType");
}

// Map a TRITONSERVER_DataType back to a DLPack dtype {code, bits, lanes=1}.
inline DLDataType from_triton_data_type(TRITONSERVER_DataType dtype)
{
  constexpr auto kInt = static_cast<uint8_t>(DLDataTypeCode::kInt);
  constexpr auto kUInt = static_cast<uint8_t>(DLDataTypeCode::kUInt);
  constexpr auto kFloat = static_cast<uint8_t>(DLDataTypeCode::kFloat);
  constexpr auto kBFloat = static_cast<uint8_t>(DLDataTypeCode::kBFloat);
  constexpr auto kBool = static_cast<uint8_t>(DLDataTypeCode::kBool);
  switch (dtype) {
    case TRITONSERVER_TYPE_INT8: return {kInt, 8, 1};
    case TRITONSERVER_TYPE_INT16: return {kInt, 16, 1};
    case TRITONSERVER_TYPE_INT32: return {kInt, 32, 1};
    case TRITONSERVER_TYPE_INT64: return {kInt, 64, 1};
    case TRITONSERVER_TYPE_UINT8: return {kUInt, 8, 1};
    case TRITONSERVER_TYPE_UINT16: return {kUInt, 16, 1};
    case TRITONSERVER_TYPE_UINT32: return {kUInt, 32, 1};
    case TRITONSERVER_TYPE_UINT64: return {kUInt, 64, 1};
    case TRITONSERVER_TYPE_FP16: return {kFloat, 16, 1};
    case TRITONSERVER_TYPE_FP32: return {kFloat, 32, 1};
    case TRITONSERVER_TYPE_FP64: return {kFloat, 64, 1};
    case TRITONSERVER_TYPE_BF16: return {kBFloat, 16, 1};
    case TRITONSERVER_TYPE_BOOL: return {kBool, 8, 1};
    default:
      throw std::invalid_argument(
              "triton_conversions: TRITONSERVER_DataType not representable as a "
              "DLPack dtype: " + std::string(TRITONSERVER_DataTypeString(dtype)));
  }
}

// Copy a Tensor's int64[] shape into a Triton shape vector.
inline std::vector<int64_t> to_triton_shape(const Tensor & tensor)
{
  return std::vector<int64_t>(tensor.shape.begin(), tensor.shape.end());
}

// Product of dimensions (element count) of a Triton shape. A dynamic dimension
// is marked with -1 (e.g. from a model config before the shape is resolved);
// throw rather than wrap the unsigned accumulator on any negative extent.
inline size_t num_elements(const std::vector<int64_t> & shape)
{
  size_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      throw std::invalid_argument(
              "triton_conversions: dynamic/negative dimension " + std::to_string(dim) +
              "; resolve the shape before computing element count");
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

namespace detail
{

template<typename T>
struct IsStdVector : std::false_type {};

template<typename T, typename AllocatorT>
struct IsStdVector<std::vector<T, AllocatorT>> : std::true_type {};

// Cast a byte pointer to void while preserving its const-ness, so the
// pointer a Triton I/O exposes stays as read-only or writable as its handle.
inline void * as_void_ptr(uint8_t * ptr) {return ptr;}
inline const void * as_void_ptr(const uint8_t * ptr) {return ptr;}

template<typename DataT>
void resize_or_allocate(DataT & data, size_t size_bytes)
{
  if constexpr (IsStdVector<DataT>::value) {
    data.resize(size_bytes);
  } else {
    data = cuda_buffer_backend::allocate_buffer(size_bytes);
  }
}

template<typename DataT>
bool is_cuda_backed(const DataT & data)
{
  if constexpr (IsStdVector<DataT>::value) {
    (void)data;
    return false;
  } else {
    return data.get_backend_type() == "cuda";
  }
}

template<typename DataT>
cuda_buffer_backend::ReadHandle make_input_handle(const DataT & data, cudaStream_t stream)
{
  if constexpr (IsStdVector<DataT>::value) {
    rosidl::Buffer<uint8_t> cpu_buffer(data.size());
    std::memcpy(cpu_buffer.data(), data.data(), data.size());
    // CPU input promotion is owned by the returned ReadHandle; the source buffer
    // only needs to stay alive until the upload has completed.
    auto read_handle = cuda_buffer_backend::from_input_buffer(cpu_buffer, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return read_handle;
  } else {
    const bool input_is_cuda_backed = is_cuda_backed(data);
    auto read_handle = cuda_buffer_backend::from_input_buffer(data, stream);
    if (!input_is_cuda_backed) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return read_handle;
  }
}

template<typename DataT>
cuda_buffer_backend::WriteHandle make_output_handle(DataT & data, cudaStream_t stream)
{
  if constexpr (IsStdVector<DataT>::value) {
    (void)data;
    (void)stream;
    throw std::runtime_error(
            "triton_conversions::from_output_tensor requires CUDA-backed tensor data; use "
            "copy_to_tensor for CPU-backed Triton outputs");
  } else {
    if (!is_cuda_backed(data)) {
      throw std::runtime_error(
              "triton_conversions::from_output_tensor requires CUDA-backed tensor data; use "
              "copy_to_tensor for CPU-backed Triton outputs");
    }
    return cuda_buffer_backend::from_output_buffer(data, stream);
  }
}

template<typename DataT>
void validate_copy_size(const DataT & data, size_t byte_offset, size_t src_size_bytes)
{
  if (byte_offset > data.size() || src_size_bytes > data.size() - byte_offset) {
    throw std::runtime_error("triton_conversions: copy exceeds tensor data buffer size");
  }
}

template<typename DataT>
void copy_device_to_data(
  DataT & data, size_t byte_offset, const void * src_device_ptr, size_t src_size_bytes,
  cudaStream_t stream)
{
  validate_copy_size(data, byte_offset, src_size_bytes);
  if constexpr (!IsStdVector<DataT>::value) {
    if (is_cuda_backed(data)) {
      auto handle = cuda_buffer_backend::from_output_buffer(data, stream);
      CUDA_CHECK(cudaMemcpyAsync(
        handle.get_ptr() + byte_offset, src_device_ptr, src_size_bytes,
        cudaMemcpyDeviceToDevice, stream));
      return;
    }
  }
  CUDA_CHECK(cudaMemcpyAsync(
    data.data() + byte_offset, src_device_ptr, src_size_bytes,
    cudaMemcpyDeviceToHost, stream));
}

template<typename DataT>
void copy_host_to_data(
  DataT & data, size_t byte_offset, const void * src_buffer, size_t src_size_bytes,
  cudaStream_t stream)
{
  validate_copy_size(data, byte_offset, src_size_bytes);
  if constexpr (!IsStdVector<DataT>::value) {
    if (is_cuda_backed(data)) {
      auto handle = cuda_buffer_backend::from_output_buffer(data, stream);
      CUDA_CHECK(cudaMemcpyAsync(
        handle.get_ptr() + byte_offset, src_buffer, src_size_bytes,
        cudaMemcpyHostToDevice, stream));
      return;
    }
  }
  (void)stream;
  std::memcpy(data.data() + byte_offset, src_buffer, src_size_bytes);
}

}  // namespace detail

// Owns the buffer Read/WriteHandle for the lifetime of the Triton I/O, and
// caches the shape/dtype/size needed to register it with an inference request.
// The wrapped buffer lives in GPU memory (TRITONSERVER_MEMORY_GPU, id 0).
template<typename HandleT>
class TritonTensor
{
public:
  TritonTensor() = default;

  TritonTensor(
    HandleT handle, std::string name, std::vector<int64_t> shape, TRITONSERVER_DataType dtype,
    size_t size_bytes, size_t byte_offset = 0)
  : handle_(std::move(handle)), name_(std::move(name)), shape_(std::move(shape)), dtype_(dtype),
    size_bytes_(size_bytes), byte_offset_(byte_offset)
  {
  }

  TritonTensor(const TritonTensor &) = delete;
  TritonTensor & operator=(const TritonTensor &) = delete;
  TritonTensor(TritonTensor &&) = default;
  TritonTensor & operator=(TritonTensor &&) = default;
  ~TritonTensor() = default;

  // Device pointer for TRITONSERVER_InferenceRequestAppendInputData, advanced by
  // the tensor's byte_offset (nonzero for a view into a larger allocation). Its
  // const-ness follows the handle: a ReadHandle (input) yields a const void*, a
  // WriteHandle (output) yields a void*. AppendInputData takes a const void*, so
  // an input binding passes through with no const_cast.
  auto data() {return detail::as_void_ptr(handle_.get_ptr() + byte_offset_);}

  const std::vector<int64_t> & shape() const {return shape_;}
  TRITONSERVER_DataType data_type() const {return dtype_;}
  size_t size_bytes() const {return size_bytes_;}

  TRITONSERVER_MemoryType memory_type() const {return TRITONSERVER_MEMORY_GPU;}
  int64_t memory_type_id() const {return 0;}

private:
  HandleT handle_;
  std::string name_{};
  std::vector<int64_t> shape_;
  TRITONSERVER_DataType dtype_{TRITONSERVER_TYPE_INVALID};
  size_t size_bytes_{0};
  size_t byte_offset_{0};
};

// Allocate a CUDA-backed Tensor sized for (shape, dtype). Fills
// name/dtype/shape/data with a contiguous row-major layout: strides is left
// empty (the DLPack convention for "contiguous, infer row-major from shape")
// and byte_offset is 0. The name is the model's I/O tensor name.
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

  const size_t num = num_elements(shape);
  const size_t size_bytes = num * element_size;
  detail::resize_or_allocate(tensor.data, size_bytes);
  return tensor;
}

// Read view over tensor.data; shape/dtype/offset derived from the message.
inline TritonTensor<cuda_buffer_backend::ReadHandle> from_input_tensor(
  std::string name, const Tensor & tensor, cudaStream_t stream)
{
  auto handle = detail::make_input_handle(tensor.data, stream);
  std::vector<int64_t> shape = to_triton_shape(tensor);
  const TRITONSERVER_DataType dtype = to_triton_data_type(
    tensor.dtype_code, tensor.dtype_bits, tensor.dtype_lanes);
  const size_t size_bytes = num_elements(shape) *
    bytes_per_element(tensor.dtype_bits, tensor.dtype_lanes);
  return TritonTensor<cuda_buffer_backend::ReadHandle>(
    std::move(handle), std::move(name), std::move(shape), dtype, size_bytes, tensor.byte_offset);
}

// Direct zero-copy write view over tensor.data; shape/dtype/offset derived from
// the message. CPU-backed outputs should use copy_to_tensor() after Triton
// returns a response so the copy-back into host storage is explicit.
inline TritonTensor<cuda_buffer_backend::WriteHandle> from_output_tensor(
  std::string name, Tensor & tensor, cudaStream_t stream)
{
  auto handle = detail::make_output_handle(tensor.data, stream);
  std::vector<int64_t> shape = to_triton_shape(tensor);
  const TRITONSERVER_DataType dtype = to_triton_data_type(
    tensor.dtype_code, tensor.dtype_bits, tensor.dtype_lanes);
  const size_t size_bytes = num_elements(shape) *
    bytes_per_element(tensor.dtype_bits, tensor.dtype_lanes);
  return TritonTensor<cuda_buffer_backend::WriteHandle>(
    std::move(handle), std::move(name), std::move(shape), dtype, size_bytes, tensor.byte_offset);
}

// Copy a standalone device buffer (e.g. a Triton inference output) into a
// pre-allocated Tensor message's buffer. `src_size_bytes` must not exceed the
// message's allocation.
inline void to_tensor(
  Tensor & tensor, const void * src_device_ptr, size_t src_size_bytes, cudaStream_t stream)
{
  try {
    detail::copy_device_to_data(tensor.data, tensor.byte_offset, src_device_ptr, src_size_bytes,
      stream);
  } catch (const cuda_buffer_backend::CudaError & err) {
    throw std::runtime_error(
            std::string("triton_conversions::to_tensor: CUDA copy failed: ") + err.what());
  }
}

inline void copy_to_tensor(
  Tensor & tensor, const void * src_buffer, size_t src_size_bytes,
  TRITONSERVER_MemoryType memory_type, cudaStream_t stream)
{
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
    to_tensor(tensor, src_buffer, src_size_bytes, stream);
    return;
  }

  try {
    detail::copy_host_to_data(tensor.data, tensor.byte_offset, src_buffer, src_size_bytes, stream);
  } catch (const cuda_buffer_backend::CudaError & err) {
    throw std::runtime_error(
            std::string("triton_conversions::copy_to_tensor: CUDA copy failed: ") + err.what());
  }
}

// Allocate a CUDA-backed Tensor for (shape, dtype) and copy a device
// buffer into it.
inline Tensor to_tensor(
  const std::vector<int64_t> & shape, uint8_t dtype_code, uint8_t dtype_bits,
  const void * src_device_ptr, cudaStream_t stream, uint16_t dtype_lanes = 1)
{
  Tensor tensor = allocate_tensor(shape, dtype_code, dtype_bits, dtype_lanes);
  const size_t size_bytes = num_elements(to_triton_shape(tensor)) *
    bytes_per_element(dtype_bits, dtype_lanes);
  to_tensor(tensor, src_device_ptr, size_bytes, stream);
  return tensor;
}

// ---------------------------------------------------------------------------
// Optional lifetime helpers: bind handle lifetime to Triton's own completion
// signals (the in-process C API's release callbacks) so the caller need not keep
// the TritonTensor holders in scope until the async inference finishes. These
// are the Triton analog of a torch::from_blob deleter. Use them instead of the
// keep-the-holder-alive pattern when you want lifetime to be correct by
// construction.
// ---------------------------------------------------------------------------

// Heap bundle of a request's input read handles. Owned by the request's release
// callback (as userp) and destroyed -- recording each input's read-completion
// event on its stream -- when Triton releases the request.
struct InputReleaseBundle
{
  std::vector<TritonTensor<cuda_buffer_backend::ReadHandle>> inputs;
};

// TRITONSERVER_InferenceRequestReleaseFn_t. Drops the InputReleaseBundle passed
// as userp once Triton is done with the request's inputs, and deletes the
// released request (the common single-use pattern; write your own callback over
// InputReleaseBundle if you pool/reuse requests).
inline void input_release_callback(
  TRITONSERVER_InferenceRequest * request, const uint32_t flags, void * userp)
{
  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    delete static_cast<InputReleaseBundle *>(userp);
    TRITONSERVER_InferenceRequestDelete(request);
  }
}

// Transfer ownership of `inputs` to `request`: they are dropped (recording their
// read-completion events) when Triton releases the request, so the caller need
// not keep the TritonTensors in scope past ServerInferAsync. Throws on the Triton
// error (and frees the bundle so it does not leak on failure).
inline void bind_inputs_to_request(
  TRITONSERVER_InferenceRequest * request,
  std::vector<TritonTensor<cuda_buffer_backend::ReadHandle>> inputs)
{
  auto * bundle = new InputReleaseBundle{std::move(inputs)};
  TRITONSERVER_Error * err = TRITONSERVER_InferenceRequestSetReleaseCallback(
    request, input_release_callback, bundle);
  if (err != nullptr) {
    delete bundle;
    const std::string msg = TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
    throw std::runtime_error(
            "triton_conversions: InferenceRequestSetReleaseCallback failed: " + msg);
  }
}

// A pre-bound output: the WriteHandle over a message buffer Triton should write
// its result into, plus that buffer's byte size for bounds checking.
struct BoundOutput
{
  cuda_buffer_backend::WriteHandle handle;
  size_t size_bytes;
};

// Per-inference table mapping a Triton output tensor name to its BoundOutput.
// Passed as the response allocator's userp so output_alloc can hand Triton the
// message's device pointer (zero-copy, no allocation) and output_release can drop
// the handle -- recording the write-completion event -- when Triton frees the
// response.
struct OutputBindingTable
{
  std::unordered_map<std::string, BoundOutput> outputs;
};

// TRITONSERVER_ResponseAllocatorAllocFn_t. Hands Triton the device pointer of the
// pre-bound output for `tensor_name` instead of allocating, moving that output's
// WriteHandle onto the heap as buffer_userp so its write-completion event is
// recorded when output_release runs.
inline TRITONSERVER_Error * output_alloc(
  TRITONSERVER_ResponseAllocator * allocator, const char * tensor_name, size_t byte_size,
  TRITONSERVER_MemoryType preferred_memory_type, int64_t preferred_memory_type_id,
  void * userp, void ** buffer, void ** buffer_userp,
  TRITONSERVER_MemoryType * actual_memory_type, int64_t * actual_memory_type_id)
{
  (void)allocator;
  (void)preferred_memory_type;
  (void)preferred_memory_type_id;
  *actual_memory_type = TRITONSERVER_MEMORY_GPU;
  *actual_memory_type_id = 0;
  *buffer = nullptr;
  *buffer_userp = nullptr;
  if (byte_size == 0) {
    return nullptr;
  }
  auto * table = static_cast<OutputBindingTable *>(userp);
  auto it = table->outputs.find(tensor_name);
  if (it == table->outputs.end()) {
    return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL,
      (std::string("triton_conversions: no pre-bound output buffer for '") +
      tensor_name + "'").c_str());
  }
  if (byte_size > it->second.size_bytes) {
    return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL,
      (std::string("triton_conversions: output '") + tensor_name + "' needs " +
      std::to_string(byte_size) + " bytes but the pre-bound buffer holds " +
      std::to_string(it->second.size_bytes)).c_str());
  }
  auto handle = std::make_unique<cuda_buffer_backend::WriteHandle>(std::move(it->second.handle));
  *buffer = handle->get_ptr();
  *buffer_userp = handle.release();
  return nullptr;
}

// TRITONSERVER_ResponseAllocatorReleaseFn_t. Destroys the WriteHandle staged by
// output_alloc (recording its write-completion event). It does NOT free the
// buffer: that memory is owned by the ROS message, not Triton.
inline TRITONSERVER_Error * output_release(
  TRITONSERVER_ResponseAllocator * allocator, void * buffer, void * buffer_userp,
  size_t byte_size, TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
{
  (void)allocator;
  (void)buffer;
  (void)byte_size;
  (void)memory_type;
  (void)memory_type_id;
  delete static_cast<cuda_buffer_backend::WriteHandle *>(buffer_userp);
  return nullptr;
}

// RAII owner of a TRITONSERVER_ResponseAllocator plus its OutputBindingTable,
// configured (via output_alloc/output_release) so Triton writes each named output
// directly into its pre-bound message buffer (zero-copy) and records that output's
// write-completion event when it frees the response. This is the output-side
// analog of bind_inputs_to_request: build it from the outputs, hand allocator()
// and table() to TRITONSERVER_InferenceRequestSetResponseCallback, and keep it
// alive until the response has been delivered.
//
// Unlike the input bundle (whose lifetime Triton owns via the request-release
// callback), Triton's response-allocator model requires the allocator and its
// userp to outlive the inference, so this holder must stay in scope until then --
// but the GPU-critical part (each output's write event) is still recorded
// automatically by output_release, not by this holder's destruction. Move-only;
// the destructor frees the allocator (the output WriteHandles are already dropped
// by output_release as Triton releases the response).
class ResponseOutputBinding
{
public:
  ResponseOutputBinding() = default;

  ResponseOutputBinding(
    TRITONSERVER_ResponseAllocator * allocator, std::unique_ptr<OutputBindingTable> table)
  : allocator_(allocator), table_(std::move(table)) {}

  ResponseOutputBinding(const ResponseOutputBinding &) = delete;
  ResponseOutputBinding & operator=(const ResponseOutputBinding &) = delete;

  ResponseOutputBinding(ResponseOutputBinding && other) noexcept
  : allocator_(other.allocator_), table_(std::move(other.table_))
  {
    other.allocator_ = nullptr;
  }

  ResponseOutputBinding & operator=(ResponseOutputBinding && other) noexcept
  {
    if (this != &other) {
      reset();
      allocator_ = other.allocator_;
      table_ = std::move(other.table_);
      other.allocator_ = nullptr;
    }
    return *this;
  }

  ~ResponseOutputBinding() {reset();}

  // The allocator to pass to TRITONSERVER_InferenceRequestSetResponseCallback.
  TRITONSERVER_ResponseAllocator * allocator() const {return allocator_;}

  // The response_allocator_userp to pass alongside allocator(); output_alloc reads
  // the pre-bound outputs from it.
  OutputBindingTable * table() const {return table_.get();}

private:
  void reset() noexcept
  {
    if (allocator_ != nullptr) {
      TRITONSERVER_Error * err = TRITONSERVER_ResponseAllocatorDelete(allocator_);
      if (err != nullptr) {TRITONSERVER_ErrorDelete(err);}
      allocator_ = nullptr;
    }
  }

  TRITONSERVER_ResponseAllocator * allocator_{nullptr};
  std::unique_ptr<OutputBindingTable> table_;
};

// Build a ResponseOutputBinding from a name -> BoundOutput map: creates the
// response allocator and moves the outputs into its table so the caller need not
// wire up TRITONSERVER_ResponseAllocatorNew / OutputBindingTable by hand. Throws
// on the Triton error.
inline ResponseOutputBinding bind_outputs(
  std::unordered_map<std::string, BoundOutput> outputs)
{
  auto table = std::make_unique<OutputBindingTable>();
  table->outputs = std::move(outputs);

  TRITONSERVER_ResponseAllocator * allocator = nullptr;
  TRITONSERVER_Error * err = TRITONSERVER_ResponseAllocatorNew(
    &allocator, output_alloc, output_release, nullptr);
  if (err != nullptr) {
    const std::string msg = TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
    throw std::runtime_error(
            "triton_conversions: ResponseAllocatorNew failed: " + msg);
  }
  return ResponseOutputBinding(allocator, std::move(table));
}

}  // namespace triton_conversions
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // TRITON_CONVERSIONS__TRITON_CONVERSIONS_HPP_
