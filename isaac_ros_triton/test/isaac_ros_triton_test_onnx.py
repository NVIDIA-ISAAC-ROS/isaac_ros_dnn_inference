# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from pathlib import Path
import shutil
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


MODEL_NAME = 'mobilenetv2-1.0_triton_onnx'
MODEL_REPOSITORY = tempfile.TemporaryDirectory()


def create_model_repository() -> str:
    """Create a Triton model repository with an unrelated non-Triton sibling."""
    model_repository = Path(MODEL_REPOSITORY.name)
    source_model = Path(__file__).resolve().parent / 'models' / MODEL_NAME
    shutil.copytree(source_model, model_repository / MODEL_NAME)

    extra_model_dir = model_repository / 'non_triton_onnx'
    extra_model_dir.mkdir()
    (extra_model_dir / 'model.onnx').write_text(
        'This file intentionally is not a valid Triton model repository entry.')

    return str(model_repository)


def get_backend_directory() -> str:
    """Return the Triton backend directory from the local build tree."""
    package_prefix = Path(get_package_prefix('isaac_ros_triton'))
    backend_directory = (
        package_prefix.parent.parent / 'build' / 'isaac_ros_triton' /
        '_deps' / 'tritonserver-src' / 'backends')
    return str(backend_directory) if backend_directory.exists() else ''


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all Triton ROS 2 nodes for testing."""
    model_dir = create_model_repository()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    triton_node = ComposableNode(
        package='isaac_ros_triton',
        name='triton',
        namespace=IsaacROSTritonNodeTest.generate_namespace(),
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': MODEL_NAME,
            'model_repository_paths': [model_dir],
            'max_batch_size': 0,
            'input_binding_names': ['data'],
            'output_binding_names': ['mobilenetv20_output_flatten0_reshape0'],
            'input_tensor_names': ['input'],
            'output_tensor_names': ['output'],
            'backend_directory': get_backend_directory(),
        }]
    )

    return IsaacROSTritonNodeTest.generate_test_description([
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
            period=5.0, actions=[launch_testing.actions.ReadyToTest()])
    ])


class IsaacROSTritonNodeTest(IsaacROSBaseTest):
    """
    Proof-of-Life Test for Isaac ROS Triton Node.

    1. Sets up ROS publisher to send zero tensors
    2. Sets up ROS subscriber to listen to output channel of TritonNode
    3. Verify received tensors are the correct dimensions
    """

    SUBSCRIBER_CHANNEL = 'tensor_sub'

    NAME = 'output'
    DTYPE_CODE = 2
    DTYPE_BITS = 32
    DTYPE_LANES = 1
    DIMENSIONS = [1, 1000]
    DATA_LENGTH = 4000
    TIMEOUT_SEC = 300

    def test_triton_node(self) -> None:
        self.node._logger.info('Starting Isaac ROS Triton Node POL Test')

        received_messages = {}

        subscriber_topic_namespace = self.generate_namespace(self.SUBSCRIBER_CHANNEL)
        test_subscribers = [
            (subscriber_topic_namespace, TensorList)
        ]

        subs = self.create_logging_subscribers(
            subscription_requests=test_subscribers,
            received_messages=received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            add_received_message_timestamps=True
        )

        self.generate_namespace_lookup(['tensor_pub'])

        tensor_pub = self.node.create_publisher(
            TensorList, self.namespaces['tensor_pub'], self.DEFAULT_QOS)

        try:
            pub_tensor_list = TensorList()
            pub_tensor = ExperimentalTensor()

            pub_tensor.dtype_code = self.DTYPE_CODE
            pub_tensor.dtype_bits = self.DTYPE_BITS
            pub_tensor.dtype_lanes = self.DTYPE_LANES
            pub_tensor.shape = [1, 3, 224, 224]
            pub_tensor.strides = []
            pub_tensor.byte_offset = 0
            pub_tensor.data = [0] * 602112

            pub_tensor_list.names = ['input']
            pub_tensor_list.tensors = [pub_tensor]

            self.node._logger.info(
                'Publishing tensors until inference response received '
                f'(timeout={self.TIMEOUT_SEC}s)')
            start_time = time.time()
            while len(received_messages.get(subscriber_topic_namespace, [])) == 0:
                if time.time() - start_time > self.TIMEOUT_SEC:
                    self.fail('Timed out waiting for inference response')
                tensor_pub.publish(pub_tensor_list)
                rclpy.spin_once(self.node, timeout_sec=0.1)
                time.sleep(1)

            self.node._logger.info(
                f'Received inference response after {time.time() - start_time:.1f}s')

            for tensor_list, _ in received_messages[subscriber_topic_namespace]:
                tensor = tensor_list.tensors[0]
                tensor_name = tensor_list.names[0]

                self.assertEqual(
                    tensor_name, self.NAME,
                    f'Unexpected tensor name, expected: {self.NAME} received: {tensor_name}'
                )
                self.assertEqual(
                    tensor.dtype_code, self.DTYPE_CODE,
                    f'Unexpected tensor dtype_code, expected: {self.DTYPE_CODE} '
                    f'received: {tensor.dtype_code}'
                )
                self.assertEqual(
                    tensor.dtype_bits, self.DTYPE_BITS,
                    f'Unexpected tensor dtype_bits, expected: {self.DTYPE_BITS} '
                    f'received: {tensor.dtype_bits}'
                )
                self.assertEqual(
                    len(tensor.data.tolist()), self.DATA_LENGTH,
                    f'Unexpected tensor length, expected: {self.DATA_LENGTH} '
                    f'received: {len(tensor.data)}'
                )
                self.assertEqual(
                    tensor.shape.tolist(), self.DIMENSIONS,
                    f'Unexpected tensor dimensions, expected: {self.DIMENSIONS} '
                    f'received: {tensor.shape}'
                )

            tensor_list, _ = received_messages[subscriber_topic_namespace][-1]
            tensor = tensor_list.tensors[0]
            tensor_name = tensor_list.names[0]
            length = len(tensor.data.tolist())
            dimensions = tensor.shape.tolist()

            self.node._logger.info(
                f'Received Tensor Properties:\n'
                f'Name: {tensor_name}\n'
                f'DType Code: {tensor.dtype_code}\n'
                f'DType Bits: {tensor.dtype_bits}\n'
                f'DType Lanes: {tensor.dtype_lanes}\n'
                f'Byte Length: {length}\n'
                f'Dimensions: {dimensions}'
            )

            self.node._logger.info('Finished Isaac ROS Triton Node POL Test')
        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.assertTrue(self.node.destroy_publisher(tensor_pub))
