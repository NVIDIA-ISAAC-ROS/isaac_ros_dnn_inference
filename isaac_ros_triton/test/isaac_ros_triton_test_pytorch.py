# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import struct
import tempfile
import time

from ament_index_python.packages import get_package_prefix
from isaac_ros_tensor_msgs.msg import TensorList
from isaac_ros_test import IsaacROSBaseTest
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch_testing

import pytest
import rclpy
from tensor_msgs.msg import ExperimentalTensor
import torch


MODEL_NAME = 'arithmetic_pytorch'
INPUT_VALUES = [1.0, -2.0, 3.5, 0.0]
EXPECTED_OUTPUT_VALUES = [3.0, -3.0, 8.0, 1.0]
MODEL_REPOSITORY = tempfile.TemporaryDirectory()


class ArithmeticModel(torch.nn.Module):
    """Small deterministic model used to verify PyTorch backend inference."""

    def forward(self, tensor):
        """Double every input value and add one."""
        return tensor * 2.0 + 1.0


def create_model_repository() -> Path:
    """Create a Triton model repository containing a traced TorchScript model."""
    model_dir = Path(MODEL_REPOSITORY.name) / MODEL_NAME
    version_dir = model_dir / '1'
    version_dir.mkdir(parents=True)

    model = torch.jit.trace(ArithmeticModel(), torch.zeros((1, 4), dtype=torch.float32))
    model.save(version_dir / 'model.pt')

    (model_dir / 'config.pbtxt').write_text(
        f"""name: "{MODEL_NAME}"
platform: "pytorch_libtorch"
max_batch_size: 0
input [
  {{
    name: "INPUT__0"
    data_type: TYPE_FP32
    dims: [ 1, 4 ]
  }}
]
output [
  {{
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [ 1, 4 ]
  }}
]
version_policy: {{
  specific {{
    versions: [ 1 ]
  }}
}}
"""
    )
    return Path(MODEL_REPOSITORY.name)


def get_backend_directory() -> str:
    """Return the Triton backend directory from the local build tree."""
    package_prefix = Path(get_package_prefix('isaac_ros_triton'))
    backend_directory = (
        package_prefix.parent.parent / 'build' / 'isaac_ros_triton' /
        '_deps' / 'tritonserver-src' / 'backends')
    return str(backend_directory) if backend_directory.exists() else ''


@pytest.mark.rostest
def generate_test_description():
    """Generate a launch description with Triton configured for PyTorch."""
    model_repository = create_model_repository()
    triton_node = ComposableNode(
        package='isaac_ros_triton',
        name='triton',
        namespace=IsaacROSTritonPyTorchTest.generate_namespace(),
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': MODEL_NAME,
            'model_repository_paths': [str(model_repository)],
            'max_batch_size': 0,
            'input_binding_names': ['INPUT__0'],
            'output_binding_names': ['OUTPUT__0'],
            'input_tensor_names': ['input'],
            'output_tensor_names': ['output'],
            'backend_directory': get_backend_directory(),
        }],
    )

    return IsaacROSTritonPyTorchTest.generate_test_description([
        ComposableNodeContainer(
            name='triton_container',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[triton_node],
            namespace='triton',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        ),
        launch.actions.TimerAction(
            period=5.0, actions=[launch_testing.actions.ReadyToTest()]),
    ])


class IsaacROSTritonPyTorchTest(IsaacROSBaseTest):
    """Verify values returned by Triton's PyTorch backend."""

    TIMEOUT_SEC = 300

    def test_pytorch_inference(self) -> None:
        """Send values through the model and verify its numerical output."""
        received_messages = {}
        subscriber_topic = self.generate_namespace('tensor_sub')
        subscriptions = self.create_logging_subscribers(
            subscription_requests=[(subscriber_topic, TensorList)],
            received_messages=received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            add_received_message_timestamps=True,
        )

        self.generate_namespace_lookup(['tensor_pub'])
        publisher = self.node.create_publisher(
            TensorList, self.namespaces['tensor_pub'], self.DEFAULT_QOS)

        input_tensor = ExperimentalTensor()
        input_tensor.dtype_code = 2
        input_tensor.dtype_bits = 32
        input_tensor.dtype_lanes = 1
        input_tensor.shape = [1, 4]
        input_tensor.strides = []
        input_tensor.byte_offset = 0
        input_tensor.data = list(struct.pack('<4f', *INPUT_VALUES))
        input_message = TensorList(names=['input'], tensors=[input_tensor])

        try:
            start_time = time.time()
            while not received_messages.get(subscriber_topic):
                if time.time() - start_time > self.TIMEOUT_SEC:
                    self.fail('Timed out waiting for PyTorch inference response')
                publisher.publish(input_message)
                rclpy.spin_once(self.node, timeout_sec=0.1)
                time.sleep(0.1)

            output_message, _ = received_messages[subscriber_topic][-1]
            self.assertEqual(len(output_message.tensors), 1)
            output_tensor = output_message.tensors[0]
            self.assertEqual(output_message.names[0], 'output')
            self.assertEqual(output_tensor.dtype_code, 2)
            self.assertEqual(output_tensor.dtype_bits, 32)
            self.assertEqual(output_tensor.dtype_lanes, 1)
            self.assertEqual(output_tensor.shape.tolist(), [1, 4])

            output_values = struct.unpack('<4f', bytes(output_tensor.data))
            for actual, expected in zip(output_values, EXPECTED_OUTPUT_VALUES):
                self.assertAlmostEqual(actual, expected, places=5)
        finally:
            [self.node.destroy_subscription(subscription) for subscription in subscriptions]
            self.assertTrue(self.node.destroy_publisher(publisher))
