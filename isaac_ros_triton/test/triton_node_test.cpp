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

#include <gmock/gmock.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "isaac_ros_triton/triton_node.hpp"
#include "rclcpp/rclcpp.hpp"

namespace
{

void DelayCudaStream(void * user_data)
{
  static_cast<void>(user_data);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

struct CudaHostDeleter
{
  void operator()(uint8_t * host_ptr) const
  {
    if (host_ptr != nullptr) {
      cudaFreeHost(host_ptr);
    }
  }
};

}  // namespace

// Objective: to cover code lines where exceptions are thrown
// Approach: send Invalid Arguments for node parameters to trigger the exception

TEST(triton_node_test, test_empty_model_name)
  {
    rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("model_name", "");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TritonNode triton_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty model_name"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(triton_node_test, test_empty_model_repository_paths)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("model_name", "dummy_name");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TritonNode triton_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty model_repository_paths"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(triton_node_test, test_empty_input_tensor_names)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("model_name", "dummy_name");
  options.append_parameter_override(
    "model_repository_paths",
    std::vector<std::string>{"dummy_path"});
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TritonNode triton_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty input_tensor_names"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(triton_node_test, test_empty_input_binding_names)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("model_name", "dummy_name");
  options.append_parameter_override(
    "model_repository_paths",
    std::vector<std::string>{"dummy_path"});
  options.append_parameter_override("input_tensor_names", std::vector<std::string>{"dummy"});
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TritonNode triton_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty input_binding_names"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(triton_node_test, test_empty_output_tensor_names)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("model_name", "dummy_name");
  options.append_parameter_override(
    "model_repository_paths",
    std::vector<std::string>{"dummy_path"});
  options.append_parameter_override("input_tensor_names", std::vector<std::string>{"dummy"});
  options.append_parameter_override("input_binding_names", std::vector<std::string>{"dummy"});
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TritonNode triton_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty output_tensor_names"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(triton_node_test, test_empty_output_binding_names)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("model_name", "dummy_name");
  options.append_parameter_override(
    "model_repository_paths",
    std::vector<std::string>{"dummy_path"});
  options.append_parameter_override("input_tensor_names", std::vector<std::string>{"dummy"});
  options.append_parameter_override("input_binding_names", std::vector<std::string>{"dummy"});
  options.append_parameter_override("output_tensor_names", std::vector<std::string>{"dummy"});
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TritonNode triton_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty output_binding_names"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(triton_node_test, prepare_input_handles_waits_for_async_buffer_writes)
{
  using nvidia::isaac_ros::common::CudaStreamPtr;
  namespace TritonConv = nvidia::isaac_ros::triton_conversions;

  CudaStreamPtr producer_stream(new cudaStream_t{nullptr});
  ASSERT_EQ(
    cudaSuccess, cudaStreamCreateWithFlags(producer_stream.get(), cudaStreamNonBlocking));
  CudaStreamPtr handoff_stream(new cudaStream_t{nullptr});
  ASSERT_EQ(
    cudaSuccess, cudaStreamCreateWithFlags(handoff_stream.get(), cudaStreamNonBlocking));
  CudaStreamPtr external_consumer_stream(new cudaStream_t{nullptr});
  ASSERT_EQ(
    cudaSuccess,
    cudaStreamCreateWithFlags(external_consumer_stream.get(), cudaStreamNonBlocking));

  uint8_t * observed_ptr{nullptr};
  ASSERT_EQ(
    cudaSuccess,
    cudaMallocHost(reinterpret_cast<void **>(&observed_ptr), sizeof(uint8_t)));
  std::unique_ptr<uint8_t, CudaHostDeleter> observed(observed_ptr);
  *observed = 0;

  {
    auto tensor = TritonConv::allocate_tensor(
      {1}, static_cast<uint8_t>(TritonConv::DLDataTypeCode::kUInt), 8);
    {
      auto write_handle = cuda_buffer_backend::from_output_buffer(
        tensor.data, *producer_stream);
      ASSERT_EQ(
        cudaSuccess,
        cudaMemsetAsync(write_handle.get_ptr(), 0, sizeof(uint8_t), *producer_stream));
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(*producer_stream));
      ASSERT_EQ(cudaSuccess, cudaLaunchHostFunc(*producer_stream, DelayCudaStream, nullptr));
      ASSERT_EQ(
        cudaSuccess,
        cudaMemsetAsync(write_handle.get_ptr(), 0x2A, sizeof(uint8_t), *producer_stream));
    }

    isaac_ros_tensor_msgs::msg::TensorList tensor_list;
    tensor_list.names.push_back("input");
    tensor_list.tensors.push_back(std::move(tensor));

    auto input_handles = nvidia::isaac_ros::dnn_inference::TritonNode::
      PrepareInputHandlesForExternalConsumer(
      tensor_list, {"input"}, *handoff_stream);

    ASSERT_EQ(
      cudaSuccess,
      cudaMemcpyAsync(
        observed.get(), input_handles.front().data(), sizeof(uint8_t),
        cudaMemcpyDeviceToHost, *external_consumer_stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(*external_consumer_stream));
    EXPECT_EQ(0x2A, *observed);
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
