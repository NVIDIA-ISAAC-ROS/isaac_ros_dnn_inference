// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_dnn_inference_test/test_tensor_publisher_node.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::executors::SingleThreadedExecutor exec;

  rclcpp::NodeOptions node_options;

  auto pub_node = std::make_shared<nvidia::isaac_ros::dnn_inference::TestTensorPublisherNode>(
    node_options);

  exec.add_node(pub_node);

  // Spin with all the components loaded
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
