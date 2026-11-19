// Copyright 2026 Meta Platforms, Inc. and affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

// Sends recorded input tensors to the inference node as TensorList messages,
// cycling through them (use 3 or more, so that a reply holding an earlier frame's
// values is caught), waits for the reply with the request's frame id, and checks
// it against the expected output for that input. A reply fails when it is late,
// lacks the output tensor, has another shape or data type, has less
// data than its shape needs, is not in a CUDA buffer when the input was, holds a
// non-finite value, or differs from the expected values by more than the
// tolerance. Exits nonzero on any failure, including bad parameters.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "executorch/extension/tensor/tensor_ptr_maker.h"
#include "executorch_conversions/executorch_conversions.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"

using executorch::aten::Device;
using executorch::aten::DeviceType;
using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::TensorPtr;
using TensorListMsg = isaac_ros_tensor_msgs::msg::TensorList;
using Clock = std::chrono::steady_clock;

namespace
{

// DLPack data type code for floating point.
constexpr uint8_t kDLPackFloat = 2;

std::vector<float> read_floats(const std::string & path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("cannot open " + path);
  }
  std::vector<float> values(static_cast<size_t>(file.tellg()) / sizeof(float));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(float));
  return values;
}

// Largest absolute difference, or infinity when any value is not finite.
double max_error(const float * got, const std::vector<float> & expected)
{
  double err = 0.0;
  for (size_t k = 0; k < expected.size(); ++k) {
    if (!std::isfinite(got[k])) {
      return INFINITY;
    }
    err = std::max(err, static_cast<double>(std::fabs(got[k] - expected[k])));
  }
  return err;
}

int run(const rclcpp::Node::SharedPtr & node)
{
  const auto input_files = node->declare_parameter<std::vector<std::string>>(
    "input_files", std::vector<std::string>{});
  const auto expected_files = node->declare_parameter<std::vector<std::string>>(
    "expected_files", std::vector<std::string>{});
  const auto shape = node->declare_parameter<std::vector<int64_t>>(
    "input_shape", std::vector<int64_t>{});
  const auto output_shape = node->declare_parameter<std::vector<int64_t>>(
    "output_shape", std::vector<int64_t>{});
  const auto input_name = node->declare_parameter<std::string>("input_tensor_name", "input_tensor");
  const auto output_name = node->declare_parameter<std::string>("output_tensor_name", "output");
  const bool on_gpu = node->declare_parameter<bool>("input_on_gpu", true);
  const auto runs = node->declare_parameter<int>("runs", 100);
  const auto tolerance = node->declare_parameter<double>("tolerance", 1e-2);
  const auto timeout = std::chrono::duration<double>(
    node->declare_parameter<double>("timeout_s", 5.0));

  if (input_files.empty() || input_files.size() != expected_files.size()) {
    throw std::invalid_argument("input_files and expected_files need the same nonzero length");
  }
  if (!std::isfinite(tolerance) || tolerance < 0) {
    throw std::invalid_argument("tolerance must be a finite number, 0 or more");
  }
  const auto numel = std::accumulate(
    shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<float>> expected;
  for (size_t k = 0; k < input_files.size(); ++k) {
    inputs.push_back(read_floats(input_files[k]));
    expected.push_back(read_floats(expected_files[k]));
    if (!std::all_of(
        expected.back().begin(), expected.back().end(), [](float v) {return std::isfinite(v);}))
    {
      throw std::invalid_argument(expected_files[k] + " holds a value that is not finite");
    }
    if (shape.empty() || static_cast<int64_t>(inputs.back().size()) != numel) {
      throw std::invalid_argument(input_files[k] + " does not hold input_shape values");
    }
    if (output_shape.empty() || static_cast<int64_t>(expected.back().size()) != std::accumulate(
        output_shape.begin(), output_shape.end(), int64_t{1}, std::multiplies<int64_t>()))
    {
      throw std::invalid_argument(expected_files[k] + " does not hold output_shape values");
    }
  }

  TensorListMsg::ConstSharedPtr reply;
  Clock::time_point received;
  rclcpp::SubscriptionOptions options;
  options.acceptable_buffer_backends = "any";
  auto subscription = node->create_subscription<TensorListMsg>(
    "tensor_sub", rclcpp::QoS(1),
    [&reply, &received](TensorListMsg::ConstSharedPtr msg) {
      reply = msg;
      received = Clock::now();
    },
    options);
  auto publisher = node->create_publisher<TensorListMsg>("tensor_pub", rclcpp::QoS(1));
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  const auto discovery_deadline = Clock::now() + std::chrono::seconds(30);
  while (rclcpp::ok() && Clock::now() < discovery_deadline &&
    (publisher->get_subscription_count() == 0 || subscription->get_publisher_count() == 0))
  {
    executor.spin_some(std::chrono::milliseconds(100));
  }
  if (publisher->get_subscription_count() == 0 || subscription->get_publisher_count() == 0) {
    RCLCPP_ERROR(node->get_logger(), "CLIENT no inference node found in 30 s result=FAIL");
    return 1;
  }

  const Device device(on_gpu ? DeviceType::CUDA : DeviceType::CPU);
  double max_err = 0.0;
  std::string input_backend;
  std::string output_backend;
  int64_t checked = 0;
  int64_t late = 0;
  int64_t failures = 0;
  int64_t malformed = 0;
  for (int64_t i = 0; i < runs && rclcpp::ok(); ++i) {
    const size_t k = static_cast<size_t>(i) % inputs.size();
    const std::string frame_id = std::to_string(i);
    // A buffer-backed message is written once, so every request gets a new one.
    auto request = std::make_unique<TensorListMsg>();
    request->header.frame_id = frame_id;
    request->names = {input_name};
    request->tensors.push_back(
      std::move(*executorch_conversions::allocate_tensor_msg(shape, ScalarType::Float, device)));
    executorch_conversions::to_tensor_msg(
      request->tensors[0], *from_blob(inputs[k].data(), {shape.begin(), shape.end()}));
    input_backend = request->tensors[0].data.get_backend_type();
    reply.reset();
    const auto sent = Clock::now();
    publisher->publish(std::move(request));
    // Replies to earlier requests are stale; keep waiting for this one.
    while (rclcpp::ok() && Clock::now() - sent < timeout &&
      (!reply || reply->header.frame_id != frame_id))
    {
      executor.spin_once(std::chrono::milliseconds(100));
    }
    // A reply that came in during the last spin, after the deadline, is late too.
    if (!reply || reply->header.frame_id != frame_id || received - sent > timeout) {
      ++late;
      ++failures;
      continue;
    }
    ++checked;
    const auto it = std::find(reply->names.begin(), reply->names.end(), output_name);
    if (it == reply->names.end() || reply->names.size() != reply->tensors.size()) {
      ++malformed;
      ++failures;
      continue;
    }
    const auto & out = reply->tensors[static_cast<size_t>(it - reply->names.begin())];
    output_backend = out.data.get_backend_type();
    // A GPU request must get a GPU reply; anything else means a copy through the CPU.
    if (on_gpu && output_backend != "cuda") {
      ++malformed;
      ++failures;
      continue;
    }
    // Checked on the message itself, so a reply of another type cannot reach the float reads.
    const bool float32 = out.dtype_code == kDLPackFloat && out.dtype_bits == 32 &&
      out.dtype_lanes == 1;
    TensorPtr view;
    if (float32 && out.shape == output_shape) {
      try {
        view = executorch_conversions::from_input_tensor_msg(out);
      } catch (const std::exception &) {
        // For example, less data than the shape needs.
      }
    }
    if (!view) {
      ++malformed;
      ++failures;
      continue;
    }
    auto host = executorch_conversions::allocate_tensor_msg(
      out.shape, ScalarType::Float, Device(DeviceType::CPU));
    executorch_conversions::to_tensor_msg(*host, *view);
    const double err = max_error(
      executorch_conversions::from_input_tensor_msg(*host)->const_data_ptr<float>(), expected[k]);
    max_err = std::max(max_err, err);
    failures += !(err <= tolerance);
  }
  const bool ok = failures == 0 && checked > 0 && checked == runs;
  RCLCPP_INFO(
    node->get_logger(),
    "CLIENT requests=%" PRId64 " checked=%" PRId64 " late=%" PRId64 " failures=%" PRId64
    " malformed=%" PRId64 " frames=%zu max_abs_err=%.6g tolerance=%.6g "
    "input_backend=%s output_backend=%s result=%s",
    static_cast<int64_t>(runs), checked, late, failures, malformed, inputs.size(), max_err,
    tolerance, input_backend.c_str(), output_backend.c_str(), ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("tensor_list_client");
  int result = 2;
  try {
    result = run(node);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(node->get_logger(), "CLIENT %s result=FAIL", error.what());
  }
  rclcpp::shutdown();
  return result;
}
