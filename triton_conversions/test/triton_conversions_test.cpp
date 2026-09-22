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

// Unit tests for triton_conversions. The DLPack dtype <-> TRITONSERVER_DataType
// maps, shape/element-count helpers, and TritonTensor accessor const-correctness
// are pure host logic (plain TEST cases). The Tensor-message <-> TritonTensor
// wiring and device copy are covered by TritonTensorTest, which builds real
// cuda_buffer Read/WriteHandles over CUDA-backed buffers and so requires a GPU.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "triton_conversions/triton_conversions.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace triton_conversions
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

TEST(TritonConversionsTest, BytesPerElementWholeByteWidths)
{
  EXPECT_EQ(bytes_per_element(8, 1), 1);
  EXPECT_EQ(bytes_per_element(16, 1), 2);
  EXPECT_EQ(bytes_per_element(32, 1), 4);
  EXPECT_EQ(bytes_per_element(64, 1), 8);
}

TEST(TritonConversionsTest, BytesPerElementHonorsLanes)
{
  EXPECT_EQ(bytes_per_element(32, 4), 16);
}

TEST(TritonConversionsTest, BytesPerElementSubByteOrZeroThrows)
{
  EXPECT_THROW(bytes_per_element(0, 1), std::invalid_argument);
  EXPECT_THROW(bytes_per_element(4, 1), std::invalid_argument);
  EXPECT_THROW(bytes_per_element(20, 1), std::invalid_argument);
}

// ============================================================================
// to_triton_data_type
// ============================================================================

TEST(TritonConversionsTest, ToTritonDataTypeIntAndUInt)
{
  EXPECT_EQ(to_triton_data_type(kInt, 8, 1), TRITONSERVER_TYPE_INT8);
  EXPECT_EQ(to_triton_data_type(kInt, 16, 1), TRITONSERVER_TYPE_INT16);
  EXPECT_EQ(to_triton_data_type(kInt, 32, 1), TRITONSERVER_TYPE_INT32);
  EXPECT_EQ(to_triton_data_type(kInt, 64, 1), TRITONSERVER_TYPE_INT64);
  EXPECT_EQ(to_triton_data_type(kUInt, 8, 1), TRITONSERVER_TYPE_UINT8);
  EXPECT_EQ(to_triton_data_type(kUInt, 16, 1), TRITONSERVER_TYPE_UINT16);
  EXPECT_EQ(to_triton_data_type(kUInt, 32, 1), TRITONSERVER_TYPE_UINT32);
  EXPECT_EQ(to_triton_data_type(kUInt, 64, 1), TRITONSERVER_TYPE_UINT64);
}

TEST(TritonConversionsTest, ToTritonDataTypeFloatFamilies)
{
  EXPECT_EQ(to_triton_data_type(kFloat, 16, 1), TRITONSERVER_TYPE_FP16);
  EXPECT_EQ(to_triton_data_type(kFloat, 32, 1), TRITONSERVER_TYPE_FP32);
  EXPECT_EQ(to_triton_data_type(kFloat, 64, 1), TRITONSERVER_TYPE_FP64);
  EXPECT_EQ(to_triton_data_type(kBFloat, 16, 1), TRITONSERVER_TYPE_BF16);
  EXPECT_EQ(to_triton_data_type(kBool, 8, 1), TRITONSERVER_TYPE_BOOL);
}

TEST(TritonConversionsTest, ToTritonDataTypeNonScalarLanesThrows)
{
  EXPECT_THROW(to_triton_data_type(kFloat, 32, 4), std::invalid_argument);
}

TEST(TritonConversionsTest, ToTritonDataTypeUnsupportedThrows)
{
  EXPECT_THROW(to_triton_data_type(kInt, 12, 1), std::invalid_argument);   // odd width
  EXPECT_THROW(to_triton_data_type(kFloat, 8, 1), std::invalid_argument);  // no fp8 in Triton
  EXPECT_THROW(to_triton_data_type(kBFloat, 32, 1), std::invalid_argument);
  EXPECT_THROW(to_triton_data_type(7, 8, 1), std::invalid_argument);       // FP8 family code
}

// ============================================================================
// from_triton_data_type
// ============================================================================

TEST(TritonConversionsTest, FromTritonDataTypeYieldsDLPackTriple)
{
  const DLDataType fp32 = from_triton_data_type(TRITONSERVER_TYPE_FP32);
  EXPECT_EQ(fp32.code, kFloat);
  EXPECT_EQ(fp32.bits, 32);
  EXPECT_EQ(fp32.lanes, 1);

  const DLDataType u8 = from_triton_data_type(TRITONSERVER_TYPE_UINT8);
  EXPECT_EQ(u8.code, kUInt);
  EXPECT_EQ(u8.bits, 8);
  EXPECT_EQ(u8.lanes, 1);

  const DLDataType bf16 = from_triton_data_type(TRITONSERVER_TYPE_BF16);
  EXPECT_EQ(bf16.code, kBFloat);
  EXPECT_EQ(bf16.bits, 16);
}

TEST(TritonConversionsTest, FromTritonDataTypeUnrepresentableThrows)
{
  EXPECT_THROW(from_triton_data_type(TRITONSERVER_TYPE_BYTES), std::invalid_argument);
  EXPECT_THROW(from_triton_data_type(TRITONSERVER_TYPE_INVALID), std::invalid_argument);
}

TEST(TritonConversionsTest, DataTypeRoundTrip)
{
  const TRITONSERVER_DataType kTypes[] = {
    TRITONSERVER_TYPE_INT8, TRITONSERVER_TYPE_INT16, TRITONSERVER_TYPE_INT32,
    TRITONSERVER_TYPE_INT64, TRITONSERVER_TYPE_UINT8, TRITONSERVER_TYPE_UINT16,
    TRITONSERVER_TYPE_UINT32, TRITONSERVER_TYPE_UINT64, TRITONSERVER_TYPE_FP16,
    TRITONSERVER_TYPE_FP32, TRITONSERVER_TYPE_FP64, TRITONSERVER_TYPE_BF16,
    TRITONSERVER_TYPE_BOOL};
  for (TRITONSERVER_DataType t : kTypes) {
    const DLDataType dt = from_triton_data_type(t);
    EXPECT_EQ(to_triton_data_type(dt.code, dt.bits, dt.lanes), t);
  }
}

// ============================================================================
// to_triton_shape / num_elements
// ============================================================================

TEST(TritonConversionsTest, ToTritonShapeCopiesDims)
{
  const std::vector<int64_t> dims{2, 3, 4};
  Tensor tensor;
  tensor.shape.assign(dims.begin(), dims.end());

  const std::vector<int64_t> shape = to_triton_shape(tensor);
  ASSERT_EQ(shape.size(), 3u);
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
  EXPECT_EQ(shape[2], 4);
}

TEST(TritonConversionsTest, ToTritonShapeScalarIsEmpty)
{
  Tensor tensor;  // rank-0 tensor: empty shape
  EXPECT_TRUE(to_triton_shape(tensor).empty());
}

TEST(TritonConversionsTest, NumElementsProduct)
{
  EXPECT_EQ(num_elements(std::vector<int64_t>{2, 3, 4}), 24u);
}

TEST(TritonConversionsTest, NumElementsEmptyIsOne)
{
  EXPECT_EQ(num_elements(std::vector<int64_t>{}), 1u);
}

TEST(TritonConversionsTest, NumElementsDynamicDimThrows)
{
  // A dynamic dimension is -1; it must not wrap the unsigned product.
  EXPECT_THROW(num_elements(std::vector<int64_t>{-1, 3, 4}), std::invalid_argument);
}

// ============================================================================
// detail::as_void_ptr — const-ness follows the pointer
// ============================================================================

TEST(TritonConversionsTest, AsVoidPtrPreservesConstness)
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
// TritonTensor — accessor const-correctness (regression for the read-handle
// const-strip) and default-constructed state.
// ============================================================================

using ReadTensor = TritonTensor<cuda_buffer_backend::ReadHandle>;
using WriteTensor = TritonTensor<cuda_buffer_backend::WriteHandle>;

TEST(TritonConversionsTest, InputTensorExposesConstPointer)
{
  static_assert(
    std::is_same<decltype(std::declval<ReadTensor &>().data()), const void *>::value,
    "input tensor data() must be const void*");
}

TEST(TritonConversionsTest, OutputTensorExposesMutablePointer)
{
  static_assert(
    std::is_same<decltype(std::declval<WriteTensor &>().data()), void *>::value,
    "output tensor data() must be void*");
}

TEST(TritonConversionsTest, DefaultConstructedTensorIsGpuMemory)
{
  ReadTensor tensor;
  EXPECT_TRUE(tensor.shape().empty());
  EXPECT_EQ(tensor.size_bytes(), 0u);
  EXPECT_EQ(tensor.memory_type(), TRITONSERVER_MEMORY_GPU);
  EXPECT_EQ(tensor.memory_type_id(), 0);
}

TEST(TritonConversionsTest, LifetimeCallbackSignaturesMatch)
{
  // The lifetime helpers are exercised end-to-end against a live server
  // (integration). Here we assert the callbacks are assignable to Triton's
  // function-pointer typedefs, i.e. their signatures are correct.
  TRITONSERVER_InferenceRequestReleaseFn_t release_fn = &input_release_callback;
  TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn = &output_alloc;
  TRITONSERVER_ResponseAllocatorReleaseFn_t out_release_fn = &output_release;
  EXPECT_NE(release_fn, nullptr);
  EXPECT_NE(alloc_fn, nullptr);
  EXPECT_NE(out_release_fn, nullptr);
}

// ============================================================================
// allocate_tensor / from_input_tensor / from_output_tensor / to_tensor --
// the Tensor-message <-> TritonTensor wiring and device copy, over real
// CUDA-backed buffers. Requires a GPU (manages a CUDA stream); tensors created
// in a test body are destroyed (releasing their handles) before TearDown
// destroys the stream.
// ============================================================================

class TritonTensorTest : public ::testing::Test
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

  std::vector<uint8_t> ReadCudaBuffer(const Tensor & tensor)
  {
    auto read_handle = cuda_buffer_backend::from_input_buffer(tensor.data, stream_);
    std::vector<uint8_t> host(tensor.data.size(), 0);
    EXPECT_EQ(
      cudaMemcpyAsync(
        host.data(), read_handle.get_ptr(), tensor.data.size(), cudaMemcpyDeviceToHost, stream_),
      cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    return host;
  }

  cudaStream_t stream_{nullptr};
};

TEST_F(TritonTensorTest, AllocateTensorSetsFieldsAndSizesBuffer)
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

TEST_F(TritonTensorTest, FromInputTensorDerivesTensorFromMessage)
{
  Tensor tensor = allocate_tensor({2, 3, 4}, kFloat, 32);
  tensor.byte_offset = 8;

  // Base device pointer of the message buffer; read handles may coexist.
  auto probe = cuda_buffer_backend::from_input_buffer(tensor.data, stream_);
  const uint8_t * base = probe.get_ptr();

  auto view = from_input_tensor("input0", tensor, stream_);

  ASSERT_EQ(view.shape().size(), 3u);
  EXPECT_EQ(view.shape()[0], 2);
  EXPECT_EQ(view.shape()[1], 3);
  EXPECT_EQ(view.shape()[2], 4);
  EXPECT_EQ(view.data_type(), TRITONSERVER_TYPE_FP32);
  EXPECT_EQ(view.size_bytes(), 24u * 4u);
  EXPECT_EQ(view.memory_type(), TRITONSERVER_MEMORY_GPU);
  EXPECT_EQ(view.memory_type_id(), 0);
  EXPECT_EQ(view.data(), static_cast<const void *>(base + 8));
}

TEST_F(TritonTensorTest, FromOutputTensorDerivesTensorFromMessage)
{
  Tensor tensor = allocate_tensor({4}, kFloat, 32);
  tensor.byte_offset = 8;

  auto view = from_output_tensor("output0", tensor, stream_);

  // A buffer allows only one write handle and forbids a write once a read exists,
  // so we capture the device base with a read handle acquired AFTER the write:
  // that read finalizes the view's write handle and returns the same base.
  auto base_handle = cuda_buffer_backend::from_input_buffer(tensor.data, stream_);
  const uint8_t * base = base_handle.get_ptr();

  ASSERT_EQ(view.shape().size(), 1u);
  EXPECT_EQ(view.shape()[0], 4);
  EXPECT_EQ(view.data_type(), TRITONSERVER_TYPE_FP32);
  EXPECT_EQ(view.size_bytes(), 4u * 4u);
  EXPECT_EQ(view.data(), static_cast<const void *>(base + 8));
}

TEST_F(TritonTensorTest, FromOutputTensorRejectsCpuBackedTensor)
{
  Tensor tensor;
  tensor.dtype_code = kUInt;
  tensor.dtype_bits = 8;
  tensor.dtype_lanes = 1;
  tensor.shape.push_back(4);
  tensor.byte_offset = 0;
  tensor.data.resize(4);

  EXPECT_FALSE(detail::is_cuda_backed(tensor.data));
  EXPECT_THROW(from_output_tensor("output0", tensor, stream_), std::runtime_error);
}

TEST_F(TritonTensorTest, ToTensorCopiesDeviceBufferIntoMessage)
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

TEST_F(TritonTensorTest, ToTensorHonorsByteOffsetForDeviceResponse)
{
  const std::vector<uint8_t> host_src{0x10, 0x20, 0x30};

  void * src = nullptr;
  ASSERT_EQ(cudaMalloc(&src, host_src.size()), cudaSuccess);
  ASSERT_EQ(
    cudaMemcpy(src, host_src.data(), host_src.size(), cudaMemcpyHostToDevice), cudaSuccess);

  Tensor tensor = allocate_tensor({6}, kUInt, 8);
  tensor.byte_offset = 2;
  to_tensor(tensor, src, host_src.size(), stream_);

  const std::vector<uint8_t> host_dst = ReadCudaBuffer(tensor);
  EXPECT_EQ(host_dst[2], host_src[0]);
  EXPECT_EQ(host_dst[3], host_src[1]);
  EXPECT_EQ(host_dst[4], host_src[2]);
  EXPECT_EQ(cudaFree(src), cudaSuccess);
}

TEST_F(TritonTensorTest, CopyToTensorHonorsByteOffsetForHostResponse)
{
  const std::vector<uint8_t> host_src{0x40, 0x50, 0x60};

  Tensor tensor = allocate_tensor({6}, kUInt, 8);
  tensor.byte_offset = 2;
  copy_to_tensor(
    tensor, host_src.data(), host_src.size(), TRITONSERVER_MEMORY_CPU, stream_);

  const std::vector<uint8_t> host_dst = ReadCudaBuffer(tensor);
  EXPECT_EQ(host_dst[2], host_src[0]);
  EXPECT_EQ(host_dst[3], host_src[1]);
  EXPECT_EQ(host_dst[4], host_src[2]);
}

// ============================================================================
// Output response-allocator callbacks and the bind_outputs helper. These drive
// output_alloc / output_release directly (as Triton would) over real WriteHandles
// -- the request-release path (input_release_callback / bind_inputs_to_request)
// needs a live server and is covered by an on-target integration test; here
// LifetimeCallbackSignaturesMatch already pins the input callback's signature.
// ============================================================================

TEST_F(TritonTensorTest, OutputAllocHandsBackBoundPointerAndReleaseKeepsBuffer)
{
  auto buffer = cuda_buffer_backend::allocate_buffer(64);
  auto wh = cuda_buffer_backend::from_output_buffer(buffer, stream_);
  void * expected = wh.get_ptr();

  OutputBindingTable table;
  table.outputs.emplace("output0", BoundOutput{std::move(wh), 64u});

  void * out_buffer = nullptr;
  void * out_userp = nullptr;
  TRITONSERVER_MemoryType actual_mt = TRITONSERVER_MEMORY_CPU;
  int64_t actual_id = -1;
  TRITONSERVER_Error * err = output_alloc(
    /*allocator=*/nullptr, "output0", /*byte_size=*/64u,
    TRITONSERVER_MEMORY_GPU, 0, &table,
    &out_buffer, &out_userp, &actual_mt, &actual_id);

  // Zero-copy: Triton is handed the message's own device pointer, staged with the
  // WriteHandle as buffer_userp; the allocation reports GPU memory.
  EXPECT_EQ(err, nullptr);
  EXPECT_EQ(out_buffer, expected);
  ASSERT_NE(out_userp, nullptr);
  EXPECT_EQ(actual_mt, TRITONSERVER_MEMORY_GPU);
  EXPECT_EQ(actual_id, 0);

  // output_release drops the staged WriteHandle (recording its write event) but
  // must NOT free the buffer -- the message owns it. Prove it stays live by
  // reading it back afterwards.
  TRITONSERVER_Error * rel = output_release(
    nullptr, out_buffer, out_userp, 64u, TRITONSERVER_MEMORY_GPU, 0);
  EXPECT_EQ(rel, nullptr);

  auto rh = cuda_buffer_backend::from_input_buffer(buffer, stream_);
  std::vector<uint8_t> host(64, 0);
  ASSERT_EQ(
    cudaMemcpyAsync(host.data(), rh.get_ptr(), 64u, cudaMemcpyDeviceToHost, stream_),
    cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

TEST_F(TritonTensorTest, OutputAllocRejectsOversizeRequest)
{
  auto buffer = cuda_buffer_backend::allocate_buffer(16);
  auto wh = cuda_buffer_backend::from_output_buffer(buffer, stream_);

  OutputBindingTable table;
  table.outputs.emplace("output0", BoundOutput{std::move(wh), 16u});

  void * out_buffer = nullptr;
  void * out_userp = nullptr;
  TRITONSERVER_MemoryType actual_mt = TRITONSERVER_MEMORY_CPU;
  int64_t actual_id = -1;
  TRITONSERVER_Error * err = output_alloc(
    nullptr, "output0", /*byte_size=*/32u,  // exceeds the 16-byte pre-bound buffer
    TRITONSERVER_MEMORY_GPU, 0, &table,
    &out_buffer, &out_userp, &actual_mt, &actual_id);

  ASSERT_NE(err, nullptr);
  TRITONSERVER_ErrorDelete(err);
  EXPECT_EQ(out_buffer, nullptr);
  EXPECT_EQ(out_userp, nullptr);
}

TEST_F(TritonTensorTest, OutputAllocRejectsUnknownTensorName)
{
  OutputBindingTable table;  // no outputs registered

  void * out_buffer = nullptr;
  void * out_userp = nullptr;
  TRITONSERVER_MemoryType actual_mt = TRITONSERVER_MEMORY_CPU;
  int64_t actual_id = -1;
  TRITONSERVER_Error * err = output_alloc(
    nullptr, "missing", 8u, TRITONSERVER_MEMORY_GPU, 0, &table,
    &out_buffer, &out_userp, &actual_mt, &actual_id);

  ASSERT_NE(err, nullptr);
  TRITONSERVER_ErrorDelete(err);
  EXPECT_EQ(out_buffer, nullptr);
}

TEST_F(TritonTensorTest, OutputAllocZeroByteReturnsNullBuffer)
{
  OutputBindingTable table;  // must not be consulted for a zero-byte output

  void * out_buffer = reinterpret_cast<void *>(0x1);
  void * out_userp = reinterpret_cast<void *>(0x1);
  TRITONSERVER_MemoryType actual_mt = TRITONSERVER_MEMORY_CPU;
  int64_t actual_id = -1;
  TRITONSERVER_Error * err = output_alloc(
    nullptr, "anything", 0u, TRITONSERVER_MEMORY_GPU, 0, &table,
    &out_buffer, &out_userp, &actual_mt, &actual_id);

  EXPECT_EQ(err, nullptr);
  EXPECT_EQ(out_buffer, nullptr);
  EXPECT_EQ(out_userp, nullptr);
}

TEST_F(TritonTensorTest, BindOutputsConfiguresAllocatorForZeroCopyOutput)
{
  auto buffer = cuda_buffer_backend::allocate_buffer(64);
  auto wh = cuda_buffer_backend::from_output_buffer(buffer, stream_);
  void * expected = wh.get_ptr();

  std::unordered_map<std::string, BoundOutput> outputs;
  outputs.emplace("output0", BoundOutput{std::move(wh), 64u});

  ResponseOutputBinding binding = bind_outputs(std::move(outputs));
  ASSERT_NE(binding.allocator(), nullptr);
  ASSERT_NE(binding.table(), nullptr);
  ASSERT_EQ(binding.table()->outputs.count("output0"), 1u);

  // Drive the configured allocator's alloc fn as Triton would, through the holder.
  void * out_buffer = nullptr;
  void * out_userp = nullptr;
  TRITONSERVER_MemoryType actual_mt = TRITONSERVER_MEMORY_CPU;
  int64_t actual_id = -1;
  TRITONSERVER_Error * err = output_alloc(
    binding.allocator(), "output0", 64u, TRITONSERVER_MEMORY_GPU, 0, binding.table(),
    &out_buffer, &out_userp, &actual_mt, &actual_id);

  EXPECT_EQ(err, nullptr);
  EXPECT_EQ(out_buffer, expected);
  ASSERT_NE(out_userp, nullptr);

  TRITONSERVER_Error * rel = output_release(
    binding.allocator(), out_buffer, out_userp, 64u, TRITONSERVER_MEMORY_GPU, 0);
  EXPECT_EQ(rel, nullptr);
  // binding's destructor frees the response allocator.
}

}  // namespace
}  // namespace triton_conversions
}  // namespace isaac_ros
}  // namespace nvidia

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
