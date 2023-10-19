# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time

from isaac_ros_tensor_list_interfaces.msg import Tensor, TensorList, TensorShape
from isaac_ros_test import IsaacROSBaseTest
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch_testing

import pytest
import rclpy


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all Triton ROS 2 nodes for testing."""
    # Loads and runs mobilenetv2-1.0
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_dir = dir_path + '/../../test/models'

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    triton_node = ComposableNode(
        package='isaac_ros_triton',
        name='triton',
        namespace=IsaacROSTritonNodeTest.generate_namespace(),
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': 'mobilenetv2-1.0_triton_onnx',
            'model_repository_paths': [model_dir],
            'max_batch_size': 0,
            'input_binding_names': ['data'],
            'output_binding_names': ['mobilenetv20_output_flatten0_reshape0'],
            'input_tensor_names': ['input'],
            'output_tensor_names': ['output']
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

    # Using default ROS-GXF Bridge output tensor channel configured in 'run_triton_inference' exe
    SUBSCRIBER_CHANNEL = 'tensor_sub'
    # The amount of seconds to allow Triton node to run before verifying received tensors
    # Will depend on time taken for Triton engine generation
    PYTHON_SUBSCRIBER_WAIT_SEC = 30.0

    # Mobilenetv2-1.0 output tensor properties to verify
    NAME = 'output'
    DATA_TYPE = 9
    DIMENSIONS = [1, 1000]
    RANK = 2
    STRIDES = [4000, 4]
    DATA_LENGTH = 4000

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
            # Create tensor compatible with mobilenetv2-1.0
            pub_tensor_list = TensorList()
            pub_tensor = Tensor()
            pub_shape = TensorShape()

            pub_shape.rank = 4
            pub_shape.dims = [1, 3, 224, 224]
            pub_tensor.shape = pub_shape

            pub_tensor.name = 'input'
            pub_tensor.data_type = self.DATA_TYPE
            pub_tensor.strides = []
            pub_tensor.data = [0] * 602112

            pub_tensor_list.tensors = [pub_tensor]

            end_time = time.time() + self.PYTHON_SUBSCRIBER_WAIT_SEC
            while time.time() < end_time:
                tensor_pub.publish(pub_tensor_list)
                time.sleep(1)
                rclpy.spin_once(self.node, timeout_sec=0.1)

            # Verify received tensors and log total number of tensors received
            num_tensors_received = len(received_messages[subscriber_topic_namespace])
            self.assertGreater(num_tensors_received, 0)
            self.node._logger.info(
                f'Received {num_tensors_received} tensors in '
                f'{self.PYTHON_SUBSCRIBER_WAIT_SEC} seconds')

            for tensor_list, _ in received_messages[subscriber_topic_namespace]:
                tensor = tensor_list.tensors[0]

                # Verify all tensor properties match that of default mobilenetv2-1.0
                self.assertEqual(
                    tensor.name, self.NAME,
                    f'Unexpected tensor name, expected: {self.NAME} received: {tensor.name}'
                )
                self.assertEqual(
                    tensor.data_type, self.DATA_TYPE,
                    f'Unexpected tensor data type, expected: {self.DATA_TYPE} '
                    f'received: {tensor.data_type}'
                )
                self.assertEqual(
                    tensor.strides.tolist(), self.STRIDES,
                    f'Unexpected tensor strides, expected: {self.STRIDES} '
                    f'received: {tensor.strides}'
                )
                self.assertEqual(
                    len(tensor.data.tolist()), self.DATA_LENGTH,
                    f'Unexpected tensor length, expected: {self.DATA_LENGTH} '
                    f'received: {len(tensor.data)}'
                )

                shape = tensor.shape

                self.assertEqual(
                    shape.rank, self.RANK,
                    f'Unexpected tensor rank, expected: {self.RANK} received: {shape.rank}'
                )
                self.assertEqual(
                    shape.dims.tolist(), self.DIMENSIONS,
                    f'Unexpected tensor dimensions, expected: {self.DIMENSIONS} '
                    f'received: {shape.dims}'
                )

            # Log properties of last received tensor
            tensor_list, _ = received_messages[subscriber_topic_namespace][-1]
            tensor = tensor_list.tensors[0]
            shape = tensor.shape
            length = len(tensor.data.tolist())
            strides = tensor.strides.tolist()
            dimensions = shape.dims.tolist()

            self.node._logger.info(
                f'Received Tensor Properties:\n'
                f'Name: {tensor.name}\n'
                f'Data Type: {tensor.data_type}\n'
                f'Strides: {strides}\n'
                f'Byte Length: {length}\n'
                f'Rank: {shape.rank}\n'
                f'Dimensions: {dimensions}'
            )

            self.node._logger.info('Finished Isaac ROS Triton Node POL Test')
        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.assertTrue(self.node.destroy_publisher(tensor_pub))
