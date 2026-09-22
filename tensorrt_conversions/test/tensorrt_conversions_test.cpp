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

// Unit tests for tensorrt_conversions. The DLPack dtype <-> nvinfer1::DataType
// maps, shape/element-count helpers, and TensorBinding accessor const-correctness
// are pure host logic (plain TEST cases). The TensorBinding bind path is covered
// by TensorRTBindingTest, which builds real cuda_buffer ReadHandle/WriteHandle
// over CUDA-backed buffers and so requires a GPU. The engine-driven allocate_/
// from_/to_ round-trips still belong to an on-target integration test.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "tensorrt_conversions/tensorrt_conversions.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace tensorrt_conversions
{
namespace
{

// DLPack dtype codes, spelled out for readability in the tests.
constexpr uint8_t kInt = static_cast<uint8_t>(DLDataTypeCode::kInt);
constexpr uint8_t kUInt = static_cast<uint8_t>(DLDataTypeCode::kUInt);
constexpr uint8_t kFloat = static_cast<uint8_t>(DLDataTypeCode::kFloat);
constexpr uint8_t kBFloat = static_cast<uint8_t>(DLDataTypeCode::kBFloat);
constexpr uint8_t kBool = static_cast<uint8_t>(DLDataTypeCode::kBool);

// ============================================================================
// bytes_per_element
// ============================================================================

TEST(TensorRTConversionsTest, BytesPerElementWholeByteWidths)
{
  EXPECT_EQ(bytes_per_element(8, 1), 1);
  EXPECT_EQ(bytes_per_element(16, 1), 2);
  EXPECT_EQ(bytes_per_element(32, 1), 4);
  EXPECT_EQ(bytes_per_element(64, 1), 8);
}

TEST(TensorRTConversionsTest, BytesPerElementSubByteOrZeroThrows)
{
  EXPECT_THROW(bytes_per_element(0, 1), std::invalid_argument);
  EXPECT_THROW(bytes_per_element(4, 1), std::invalid_argument);
}

// ============================================================================
// to_trt_data_type / from_trt_data_type
// ============================================================================

TEST(TensorRTConversionsTest, ToTrtDataTypeSupported)
{
  EXPECT_EQ(to_trt_data_type(kInt, 8, 1), nvinfer1::DataType::kINT8);
  EXPECT_EQ(to_trt_data_type(kInt, 32, 1), nvinfer1::DataType::kINT32);
  EXPECT_EQ(to_trt_data_type(kInt, 64, 1), nvinfer1::DataType::kINT64);
  EXPECT_EQ(to_trt_data_type(kUInt, 8, 1), nvinfer1::DataType::kUINT8);
  EXPECT_EQ(to_trt_data_type(kFloat, 16, 1), nvinfer1::DataType::kHALF);
  EXPECT_EQ(to_trt_data_type(kFloat, 32, 1), nvinfer1::DataType::kFLOAT);
  EXPECT_EQ(to_trt_data_type(kBFloat, 16, 1), nvinfer1::DataType::kBF16);
  EXPECT_EQ(to_trt_data_type(kBool, 8, 1), nvinfer1::DataType::kBOOL);
}

TEST(TensorRTConversionsTest, ToTrtDataTypeUnrepresentableThrows)
{
  // TensorRT has no int16/uint16/uint32/uint64/float64 binding types.
  EXPECT_THROW(to_trt_data_type(kInt, 16, 1), std::invalid_argument);
  EXPECT_THROW(to_trt_data_type(kUInt, 16, 1), std::invalid_argument);
  EXPECT_THROW(to_trt_data_type(kUInt, 32, 1), std::invalid_argument);
  EXPECT_THROW(to_trt_data_type(kUInt, 64, 1), std::invalid_argument);
  EXPECT_THROW(to_trt_data_type(kFloat, 64, 1), std::invalid_argument);
}

TEST(TensorRTConversionsTest, ToTrtDataTypeNonScalarLanesThrows)
{
  EXPECT_THROW(to_trt_data_type(kFloat, 32, 4), std::invalid_argument);
}

TEST(TensorRTConversionsTest, FromTrtDataTypeYieldsDLPackTriple)
{
  const DLDataType f32 = from_trt_data_type(nvinfer1::DataType::kFLOAT);
  EXPECT_EQ(f32.code, kFloat);
  EXPECT_EQ(f32.bits, 32);
  EXPECT_EQ(f32.lanes, 1);

  const DLDataType i64 = from_trt_data_type(nvinfer1::DataType::kINT64);
  EXPECT_EQ(i64.code, kInt);
  EXPECT_EQ(i64.bits, 64);

  const DLDataType bf16 = from_trt_data_type(nvinfer1::DataType::kBF16);
  EXPECT_EQ(bf16.code, kBFloat);
  EXPECT_EQ(bf16.bits, 16);
}

TEST(TensorRTConversionsTest, FromTrtDataTypeUnrepresentableThrows)
{
  // kFP8 has no whole-byte DLPack dtype in the supported set.
  EXPECT_THROW(from_trt_data_type(nvinfer1::DataType::kFP8), std::invalid_argument);
}

TEST(TensorRTConversionsTest, DataTypeRoundTrip)
{
  const nvinfer1::DataType kTypes[] = {
    nvinfer1::DataType::kINT8, nvinfer1::DataType::kINT32, nvinfer1::DataType::kINT64,
    nvinfer1::DataType::kUINT8, nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT,
    nvinfer1::DataType::kBF16, nvinfer1::DataType::kBOOL};
  for (nvinfer1::DataType t : kTypes) {
    const DLDataType dt = from_trt_data_type(t);
    EXPECT_EQ(to_trt_data_type(dt.code, dt.bits, dt.lanes), t);
  }
}

// ============================================================================
// to_trt_dims
// ============================================================================

TEST(TensorRTConversionsTest, ToTrtDimsCopiesShape)
{
  const std::vector<int64_t> dims{2, 3, 4};
  Tensor tensor;
  tensor.shape.assign(dims.begin(), dims.end());

  const nvinfer1::Dims trt_dims = to_trt_dims(tensor);
  ASSERT_EQ(trt_dims.nbDims, 3);
  EXPECT_EQ(trt_dims.d[0], 2);
  EXPECT_EQ(trt_dims.d[1], 3);
  EXPECT_EQ(trt_dims.d[2], 4);
}

TEST(TensorRTConversionsTest, ToTrtDimsScalarRankZero)
{
  Tensor tensor;  // empty shape
  const nvinfer1::Dims trt_dims = to_trt_dims(tensor);
  EXPECT_EQ(trt_dims.nbDims, 0);
}

TEST(TensorRTConversionsTest, ToTrtDimsRankExceedingMaxThrows)
{
  Tensor tensor;
  const std::vector<int64_t> too_many(nvinfer1::Dims::MAX_DIMS + 1, 1);
  tensor.shape.assign(too_many.begin(), too_many.end());
  EXPECT_THROW(to_trt_dims(tensor), std::invalid_argument);
}

// ============================================================================
// num_elements
// ============================================================================

TEST(TensorRTConversionsTest, NumElementsProduct)
{
  nvinfer1::Dims dims{};
  dims.nbDims = 3;
  dims.d[0] = 2;
  dims.d[1] = 3;
  dims.d[2] = 4;
  EXPECT_EQ(num_elements(dims), 24u);
}

TEST(TensorRTConversionsTest, NumElementsEmptyIsOne)
{
  nvinfer1::Dims dims{};
  dims.nbDims = 0;
  EXPECT_EQ(num_elements(dims), 1u);
}

TEST(TensorRTConversionsTest, NumElementsDynamicDimThrows)
{
  // TensorRT marks a dynamic (e.g. batch) dimension with -1; it must not wrap
  // the unsigned product into a bogus size.
  nvinfer1::Dims dims{};
  dims.nbDims = 2;
  dims.d[0] = -1;
  dims.d[1] = 3;
  EXPECT_THROW(num_elements(dims), std::invalid_argument);
}

// ============================================================================
// detail::as_void_ptr — const-ness follows the pointer
// ============================================================================

TEST(TensorRTConversionsTest, AsVoidPtrPreservesConstness)
{
  static_assert(
    std::is_same<decltype(detail::as_void_ptr(std::declval<uint8_t *>())), void *>::value,
    "mutable byte pointer must yield void*");
  static_assert(
    std::is_same<decltype(detail::as_void_ptr(std::declval<const uint8_t *>())),
    const void *>::value,
    "const byte pointer must yield const void*");
}

// ============================================================================
// TensorBinding — accessor const-correctness (regression for the read-handle
// const-strip) and default-constructed state.
// ============================================================================

using ReadBinding = TensorBinding<cuda_buffer_backend::ReadHandle>;
using WriteBinding = TensorBinding<cuda_buffer_backend::WriteHandle>;

TEST(TensorRTConversionsTest, InputBindingExposesConstPointer)
{
  static_assert(
    std::is_same<decltype(std::declval<ReadBinding &>().data()), const void *>::value,
    "input binding data() must be const void*");
}

TEST(TensorRTConversionsTest, OutputBindingExposesMutablePointer)
{
  static_assert(
    std::is_same<decltype(std::declval<WriteBinding &>().data()), void *>::value,
    "output binding data() must be void*");
}

TEST(TensorRTConversionsTest, DefaultConstructedBindingIsEmpty)
{
  ReadBinding binding;
  EXPECT_EQ(binding.dims().nbDims, 0);
  EXPECT_EQ(binding.size_bytes(), 0u);
}

// ============================================================================
// TensorBinding bind path — detail::bind_input / bind_output (which as_input and
// as_output forward to) exercised against a recording context, over real
// cuda_buffer ReadHandle/WriteHandle.
//
// A real nvinfer1::IExecutionContext needs a deserialized TensorRT engine (not
// just a GPU) and its setInputShape/setTensorAddress are non-virtual, so it
// cannot be faked by subclassing. bind_input/bind_output are templated on the
// context type, letting us drive them with the RecordingContext double and
// observe exactly what they bind. The bindings themselves use real handles
// acquired from CUDA-backed buffers, so data() returns a genuine device pointer.
// ============================================================================

// Records every setInputShape/setTensorAddress call and lets a test force either
// to report failure, mirroring nvinfer1::IExecutionContext's signatures.
class RecordingContext
{
public:
  bool set_input_shape_result = true;
  bool set_tensor_address_result = true;

  bool enqueue_v3_result = true;

  int set_input_shape_calls = 0;
  int set_tensor_address_calls = 0;
  int enqueue_v3_calls = 0;
  std::string last_shape_name;
  nvinfer1::Dims last_shape_dims{};
  std::string last_address_name;
  const void * last_address = nullptr;
  cudaStream_t last_stream = nullptr;

  bool setInputShape(const char * name, const nvinfer1::Dims & dims)
  {
    ++set_input_shape_calls;
    last_shape_name = name;
    last_shape_dims = dims;
    return set_input_shape_result;
  }

  bool setTensorAddress(const char * name, void * data)
  {
    ++set_tensor_address_calls;
    last_address_name = name;
    last_address = data;
    return set_tensor_address_result;
  }

  bool enqueueV3(cudaStream_t stream)
  {
    ++enqueue_v3_calls;
    last_stream = stream;
    return enqueue_v3_result;
  }
};

nvinfer1::Dims make_dims2(int64_t d0, int64_t d1)
{
  nvinfer1::Dims dims{};
  dims.nbDims = 2;
  dims.d[0] = d0;
  dims.d[1] = d1;
  return dims;
}

// Manages a CUDA stream for tests that acquire real cuda_buffer handles. Requires
// a GPU; bindings created in a test body are destroyed (releasing their handles)
// before TearDown destroys the stream.
class TensorRTBindingTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  }

  void TearDown() override
  {
    cudaStreamDestroy(stream_);
  }

  cudaStream_t stream_{nullptr};
};

TEST_F(TensorRTBindingTest, BindInputSetsShapeThenAddress)
{
  auto buffer = cuda_buffer_backend::allocate_buffer(64);
  auto handle = cuda_buffer_backend::from_input_buffer(buffer, stream_);
  const void * device_ptr = handle.get_ptr();
  ReadBinding binding(
    std::move(handle), "input0", make_dims2(3, 4), nvinfer1::DataType::kFLOAT,
    /*size_bytes=*/48u);

  RecordingContext context;
  detail::bind_input(context, "input0", binding.dims(), binding.data());

  EXPECT_EQ(context.set_input_shape_calls, 1);
  EXPECT_EQ(context.last_shape_name, "input0");
  EXPECT_EQ(context.last_shape_dims.nbDims, 2);
  EXPECT_EQ(context.last_shape_dims.d[0], 3);
  EXPECT_EQ(context.last_shape_dims.d[1], 4);

  EXPECT_EQ(context.set_tensor_address_calls, 1);
  EXPECT_EQ(context.last_address_name, "input0");
  EXPECT_EQ(context.last_address, device_ptr);
}

TEST_F(TensorRTBindingTest, BindInputThrowsWhenSetInputShapeFails)
{
  auto buffer = cuda_buffer_backend::allocate_buffer(16);
  auto handle = cuda_buffer_backend::from_input_buffer(buffer, stream_);
  ReadBinding binding(
    std::move(handle), "input0", make_dims2(2, 2), nvinfer1::DataType::kFLOAT, 16u);

  RecordingContext context;
  context.set_input_shape_result = false;

  EXPECT_THROW(
    detail::bind_input(context, "input0", binding.dims(), binding.data()),
    std::runtime_error);
  // A failed setInputShape must short-circuit before binding the address.
  EXPECT_EQ(context.set_input_shape_calls, 1);
  EXPECT_EQ(context.set_tensor_address_calls, 0);
}

TEST_F(TensorRTBindingTest, BindInputThrowsWhenSetTensorAddressFails)
{
  auto buffer = cuda_buffer_backend::allocate_buffer(16);
  auto handle = cuda_buffer_backend::from_input_buffer(buffer, stream_);
  ReadBinding binding(
    std::move(handle), "input0", make_dims2(2, 2), nvinfer1::DataType::kFLOAT, 16u);

  RecordingContext context;
  context.set_tensor_address_result = false;

  EXPECT_THROW(
    detail::bind_input(context, "input0", binding.dims(), binding.data()),
    std::runtime_error);
  EXPECT_EQ(context.set_input_shape_calls, 1);
  EXPECT_EQ(context.set_tensor_address_calls, 1);
}

TEST_F(TensorRTBindingTest, BindOutputSetsAddressWithoutShape)
{
  auto buffer = cuda_buffer_backend::allocate_buffer(64);
  auto handle = cuda_buffer_backend::from_output_buffer(buffer, stream_);
  void * device_ptr = handle.get_ptr();
  WriteBinding binding(
    std::move(handle), "output0", make_dims2(3, 4), nvinfer1::DataType::kFLOAT,
    /*size_bytes=*/48u);

  RecordingContext context;
  detail::bind_output(context, "output0", binding.data());

  // The engine determines an output's shape, so bind_output only sets the address.
  EXPECT_EQ(context.set_input_shape_calls, 0);
  EXPECT_EQ(context.set_tensor_address_calls, 1);
  EXPECT_EQ(context.last_address_name, "output0");
  EXPECT_EQ(context.last_address, device_ptr);
}

TEST_F(TensorRTBindingTest, BindOutputThrowsWhenSetTensorAddressFails)
{
  auto buffer = cuda_buffer_backend::allocate_buffer(16);
  auto handle = cuda_buffer_backend::from_output_buffer(buffer, stream_);
  WriteBinding binding(
    std::move(handle), "output0", make_dims2(1, 1), nvinfer1::DataType::kFLOAT, 4u);

  RecordingContext context;
  context.set_tensor_address_result = false;

  EXPECT_THROW(
    detail::bind_output(context, "output0", binding.data()), std::runtime_error);
  EXPECT_EQ(context.set_tensor_address_calls, 1);
}

// as_input/as_output feed data() into the bind_ helpers; verify data() applies
// the tensor's byte_offset to the real handle's device pointer for read and write.
TEST_F(TensorRTBindingTest, InputBindingDataAppliesByteOffset)
{
  auto buffer = cuda_buffer_backend::allocate_buffer(64);
  auto handle = cuda_buffer_backend::from_input_buffer(buffer, stream_);
  const uint8_t * base = handle.get_ptr();
  ReadBinding binding(
    std::move(handle), "input0", make_dims2(1, 1), nvinfer1::DataType::kFLOAT,
    /*size_bytes=*/4u, /*byte_offset=*/16u);
  EXPECT_EQ(binding.data(), static_cast<const void *>(base + 16));
}

TEST_F(TensorRTBindingTest, OutputBindingDataAppliesByteOffset)
{
  auto buffer = cuda_buffer_backend::allocate_buffer(64);
  auto handle = cuda_buffer_backend::from_output_buffer(buffer, stream_);
  uint8_t * base = handle.get_ptr();
  WriteBinding binding(
    std::move(handle), "output0", make_dims2(1, 1), nvinfer1::DataType::kFLOAT,
    /*size_bytes=*/4u, /*byte_offset=*/8u);
  EXPECT_EQ(binding.data(), static_cast<void *>(base + 8));
}

TEST_F(TensorRTBindingTest, EnqueueRunsOnStream)
{
  RecordingContext context;
  detail::enqueue(context, stream_);

  EXPECT_EQ(context.enqueue_v3_calls, 1);
  EXPECT_EQ(context.last_stream, stream_);
}

TEST_F(TensorRTBindingTest, EnqueueThrowsWhenEnqueueV3Fails)
{
  RecordingContext context;
  context.enqueue_v3_result = false;

  EXPECT_THROW(detail::enqueue(context, stream_), std::runtime_error);
  EXPECT_EQ(context.enqueue_v3_calls, 1);
}

// ============================================================================
// allocate_tensor / from_input_tensor / from_output_tensor / to_tensor --
// the Tensor-message <-> TensorBinding wiring and device copy, over real
// CUDA-backed buffers.
// ============================================================================

TEST_F(TensorRTBindingTest, AllocateTensorSetsFieldsAndSizesBuffer)
{
  const std::vector<int64_t> shape{2, 3, 4};
  Tensor tensor = allocate_tensor(shape, kFloat, 32);

  EXPECT_EQ(tensor.dtype_code, kFloat);
  EXPECT_EQ(tensor.dtype_bits, 32);
  EXPECT_EQ(tensor.dtype_lanes, 1);
  ASSERT_EQ(tensor.shape.size(), 3u);
  EXPECT_EQ(tensor.shape[0], 2);
  EXPECT_EQ(tensor.shape[1], 3);
  EXPECT_EQ(tensor.shape[2], 4);
  EXPECT_EQ(tensor.byte_offset, 0u);
  // 2*3*4 elements * 4 bytes/float, in a CUDA-backed buffer.
  EXPECT_EQ(tensor.data.size(), 24u * 4u);
  EXPECT_EQ(tensor.data.get_backend_type(), "cuda");
}

TEST_F(TensorRTBindingTest, FromInputTensorDerivesBindingFromMessage)
{
  Tensor tensor = allocate_tensor({2, 3, 4}, kFloat, 32);
  tensor.byte_offset = 8;

  // Base device pointer of the message buffer; read handles may coexist.
  auto probe = cuda_buffer_backend::from_input_buffer(tensor.data, stream_);
  const uint8_t * base = probe.get_ptr();

  auto binding = from_input_tensor("input0", tensor, stream_);

  ASSERT_EQ(binding.dims().nbDims, 3);
  EXPECT_EQ(binding.dims().d[0], 2);
  EXPECT_EQ(binding.dims().d[1], 3);
  EXPECT_EQ(binding.dims().d[2], 4);
  EXPECT_EQ(binding.data_type(), nvinfer1::DataType::kFLOAT);
  EXPECT_EQ(binding.size_bytes(), 24u * 4u);
  EXPECT_EQ(binding.data(), static_cast<const void *>(base + 8));
}

TEST_F(TensorRTBindingTest, FromOutputTensorDerivesBindingFromMessage)
{
  Tensor tensor = allocate_tensor({4}, kFloat, 32);
  tensor.byte_offset = 8;

  auto binding = from_output_tensor("output0", tensor, stream_);

  // A buffer allows only one write handle and forbids a write once a read exists,
  // so we capture the device base with a read handle acquired AFTER the write:
  // that read finalizes the binding's write handle and returns the same base.
  auto base_handle = cuda_buffer_backend::from_input_buffer(tensor.data, stream_);
  const uint8_t * base = base_handle.get_ptr();

  ASSERT_EQ(binding.dims().nbDims, 1);
  EXPECT_EQ(binding.dims().d[0], 4);
  EXPECT_EQ(binding.data_type(), nvinfer1::DataType::kFLOAT);
  EXPECT_EQ(binding.size_bytes(), 4u * 4u);
  EXPECT_EQ(binding.data(), static_cast<const void *>(base + 8));
}

TEST_F(TensorRTBindingTest, ToTensorCopiesDeviceBufferIntoMessage)
{
  const std::vector<float> host_src{1.5f, -2.0f, 3.25f, 4.0f};
  const size_t bytes = host_src.size() * sizeof(float);

  void * src = nullptr;
  ASSERT_EQ(cudaMalloc(&src, bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(src, host_src.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

  Tensor tensor = allocate_tensor({4}, kFloat, 32);
  to_tensor(tensor, src, bytes, stream_);

  // Read the message's device buffer back to host and verify the copy landed.
  auto rh = cuda_buffer_backend::from_input_buffer(tensor.data, stream_);
  std::vector<float> host_dst(host_src.size(), 0.0f);
  ASSERT_EQ(
    cudaMemcpyAsync(host_dst.data(), rh.get_ptr(), bytes, cudaMemcpyDeviceToHost, stream_),
    cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (size_t i = 0; i < host_src.size(); ++i) {
    EXPECT_FLOAT_EQ(host_dst[i], host_src[i]);
  }
  EXPECT_EQ(cudaFree(src), cudaSuccess);
}

TEST(TensorRTConversionsTest, AllocateTensorPopulatesDLPackFields)
{
  const std::vector<int64_t> shape{1, 3, 224, 224};
  const uint8_t dtype_code = kFloat;
  const uint8_t dtype_bits = 32;
  const uint16_t dtype_lanes = 1;

  Tensor tensor = allocate_tensor(shape, dtype_code, dtype_bits, dtype_lanes);

  EXPECT_EQ(tensor.dtype_code, dtype_code);
  EXPECT_EQ(tensor.dtype_bits, dtype_bits);
  EXPECT_EQ(tensor.dtype_lanes, dtype_lanes);
  EXPECT_EQ(tensor.shape, shape);
  EXPECT_TRUE(tensor.strides.empty());
  EXPECT_EQ(tensor.byte_offset, 0u);
}

}  // namespace
}  // namespace tensorrt_conversions
}  // namespace isaac_ros
}  // namespace nvidia

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
