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

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "executorch/extension/cuda/caller_stream.h"
#include "executorch/extension/module/module.h"
#include "executorch_conversions/executorch_conversions.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "isaac_ros_tensor_msgs/tensor_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

using executorch::aten::Device;
using executorch::aten::DeviceType;
using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::cuda::CallerStreamGuard;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;
using executorch::runtime::to_string;
using TensorListMsg = isaac_ros_tensor_msgs::msg::TensorList;
using tensor_msgs::msg::ExperimentalTensor;

namespace
{

struct TensorSpec
{
  std::vector<int64_t> shape;
  ScalarType dtype;
  Device device{DeviceType::CPU};
};

TensorSpec to_spec(
  const Result<TensorInfo> & meta,
  const std::string & what)
{
  if (!meta.ok()) {
    throw std::runtime_error(what + " is not a tensor");
  }
  // The node hands message storage to the program, so the program must not plan it.
  if (meta->is_memory_planned()) {
    throw std::runtime_error(
            what + " is memory planned; export with "
            "MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False) "
            "as the README shows");
  }
  // Messages carry row major data, so another layout would be read and written wrongly.
  const auto dim_order = meta->dim_order();
  for (size_t d = 0; d < dim_order.size(); ++d) {
    if (static_cast<size_t>(dim_order[d]) != d) {
      throw std::runtime_error(
              what + " is not in row major dim order (for example channels last); "
              "export it contiguous");
    }
  }
  return {{meta->sizes().begin(), meta->sizes().end()}, meta->scalar_type(), meta->device()};
}

}  // namespace

// Runs an ExecuTorch program on Isaac ROS TensorList messages. The topics, the
// tensor name parameters and the QoS parameters match isaac_ros_tensor_rt's TensorRTNode.
class ExecuTorchNode : public rclcpp::Node
{
public:
  explicit ExecuTorchNode(const rclcpp::NodeOptions & options)
  : Node("executorch_node", options),
    input_names_(declare_parameter<std::vector<std::string>>(
        "input_tensor_names", std::vector<std::string>{"input_tensor"})),
    output_names_(declare_parameter<std::vector<std::string>>(
        "output_tensor_names", std::vector<std::string>{"output"}))
  {
    const auto model = declare_parameter<std::string>("model_file_path", "model.pte");
    // The export writes a .ptd file for its CUDA backend parts, and the program needs it.
    const auto data_files = declare_parameter<std::vector<std::string>>(
      "data_file_paths", std::vector<std::string>{});
    const auto input_queue_size = declare_parameter<int16_t>("input_queue_size", 1);
    const auto output_queue_size = declare_parameter<int16_t>("output_queue_size", 1);
    const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
      *this, "DEFAULT", "input_qos").keep_last(input_queue_size);
    const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
      *this, "DEFAULT", "output_qos").keep_last(output_queue_size);

    // One stream for every frame: in a multithreaded container the callbacks run on
    // different threads, and each thread's default stream would let frames overlap
    // on the same program memory. A program that only runs on the CPU needs no GPU.
    int devices = 0;
    if (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0) {
      cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ExecuTorchNode");
    }

    module_ = std::make_unique<Module>(model, data_files);
    const auto meta = module_->method_meta("forward");
    if (!meta.ok()) {
      throw std::runtime_error(
              "cannot read the forward method of " + model + ": " + to_string(meta.error()));
    }
    if (meta->num_inputs() != input_names_.size() ||
      meta->num_outputs() != output_names_.size())
    {
      throw std::runtime_error(
              "the program takes " + std::to_string(meta->num_inputs()) + " inputs and " +
              std::to_string(meta->num_outputs()) + " outputs, but the parameters name " +
              std::to_string(input_names_.size()) + " and " +
              std::to_string(output_names_.size()));
    }
    for (size_t i = 0; i < meta->num_inputs(); ++i) {
      inputs_.push_back(to_spec(meta->input_tensor_meta(i), "input " + std::to_string(i)));
    }
    for (size_t i = 0; i < meta->num_outputs(); ++i) {
      outputs_.push_back(to_spec(meta->output_tensor_meta(i), "output " + std::to_string(i)));
    }
    const auto on_host = [](const TensorSpec & spec) {
        return spec.device.type() == DeviceType::CPU;
      };
    if (cuda_stream_ && (std::any_of(inputs_.begin(), inputs_.end(), on_host) ||
      std::any_of(outputs_.begin(), outputs_.end(), on_host)))
    {
      RCLCPP_WARN(
        get_logger(), "the program keeps inputs or outputs in host memory, so messages in CUDA "
        "buffers are copied through the CPU; the README shows how to export them on the GPU");
    }
    const Error loaded = module_->load_method("forward");
    if (loaded != Error::Ok) {
      throw std::runtime_error(
              "cannot load the forward method of " + model + ": " + to_string(loaded));
    }

    rclcpp::SubscriptionOptions sub_options;
    sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    sub_options.acceptable_buffer_backends = "any";
    rclcpp::PublisherOptions pub_options;
    pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    publisher_ = create_publisher<TensorListMsg>("tensor_sub", output_qos, pub_options);
    subscription_ = create_subscription<TensorListMsg>(
      "tensor_pub", input_qos,
      [this](TensorListMsg::ConstSharedPtr msg) {
        try {
          infer(*msg);
        } catch (const std::exception & error) {
          RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "skipping a message: %s",
          error.what());
        }
      },
      sub_options);
    RCLCPP_INFO(
      get_logger(), "loaded %s: %zu inputs, %zu outputs", model.c_str(), inputs_.size(),
      outputs_.size());
  }

private:
  const ExperimentalTensor & find_input(const TensorListMsg & msg, size_t i) const
  {
    const auto * tensor = isaac_ros_tensor_msgs::FindTensorByName(msg, input_names_[i]);
    if (!tensor) {
      throw std::invalid_argument("no tensor named " + input_names_[i]);
    }
    if (tensor->shape != inputs_[i].shape ||
      executorch_conversions::scalar_type(*tensor) != inputs_[i].dtype)
    {
      throw std::invalid_argument(input_names_[i] + " has another shape or dtype");
    }
    return *tensor;
  }

  void infer(const TensorListMsg & msg)
  {
    void * stream = cuda_stream_ ? *cuda_stream_ : nullptr;
    auto reply = std::make_unique<TensorListMsg>();
    reply->header = msg.header;
    reply->names = output_names_;
    reply->tensors.reserve(outputs_.size());
    for (const auto & spec : outputs_) {
      reply->tensors.push_back(
        std::move(*executorch_conversions::allocate_tensor_msg(
          spec.shape, spec.dtype, spec.device)));
    }
    {
      // A message whose storage is on another device than the program expects is
      // copied once into a message on the right device.
      std::vector<std::unique_ptr<ExperimentalTensor>> staging;
      std::vector<TensorPtr> views;
      std::vector<EValue> inputs;
      for (size_t i = 0; i < inputs_.size(); ++i) {
        auto view = executorch_conversions::from_input_tensor_msg(find_input(msg, i), stream);
        if (!view) {
          throw std::invalid_argument(input_names_[i] + " has no data");
        }
        if (view->device() != inputs_[i].device) {
          staging.push_back(
            executorch_conversions::allocate_tensor_msg(
              inputs_[i].shape, inputs_[i].dtype, inputs_[i].device));
          executorch_conversions::to_tensor_msg(*staging.back(), *view, stream);
          view = executorch_conversions::from_input_tensor_msg(*staging.back(), stream);
        }
        views.push_back(std::move(view));
        inputs.emplace_back(*views.back());
      }
      // The program writes its outputs straight into the reply's storage.
      std::vector<TensorPtr> output_views;
      for (size_t i = 0; i < outputs_.size(); ++i) {
        output_views.push_back(
          executorch_conversions::from_output_tensor_msg(reply->tensors[i], stream));
        if (!output_views.back()) {
          throw std::runtime_error("output " + std::to_string(i) + " has no storage");
        }
        const Error set = module_->set_output("forward", *output_views.back(), i);
        if (set != Error::Ok) {
          throw std::runtime_error(
                  "set_output failed for output " + std::to_string(i) + ": " + to_string(set));
        }
      }
      std::optional<CallerStreamGuard> stream_guard;
      if (cuda_stream_) {
        stream_guard.emplace(*cuda_stream_);
      }
      const auto result = module_->execute("forward", inputs);
      if (!result.ok()) {
        throw std::runtime_error(std::string("execute failed: ") + to_string(result.error()));
      }
      // An output that aliases an input or another output is not written to the reply,
      // and an output whose size depends on the data fills only part of it.
      for (size_t i = 0; i < outputs_.size(); ++i) {
        if (!result->at(i).isTensor() ||
          result->at(i).toTensor().const_data_ptr() != output_views[i]->const_data_ptr() ||
          !std::equal(
            outputs_[i].shape.begin(), outputs_[i].shape.end(),
            result->at(i).toTensor().sizes().begin(), result->at(i).toTensor().sizes().end()))
        {
          throw std::runtime_error(
                  "output " + std::to_string(i) +
                  " is not in the reply's buffer or does not have its fixed shape");
        }
      }
    }  // Releasing the views records the write events before the reply is published.
    publisher_->publish(std::move(reply));
  }

  const std::vector<std::string> input_names_;
  const std::vector<std::string> output_names_;
  // Declared before module_ so that it is destroyed after the program.
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  std::unique_ptr<Module> module_;
  std::vector<TensorSpec> inputs_;
  std::vector<TensorSpec> outputs_;
  rclcpp::Publisher<TensorListMsg>::SharedPtr publisher_;
  rclcpp::Subscription<TensorListMsg>::SharedPtr subscription_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::ExecuTorchNode)
