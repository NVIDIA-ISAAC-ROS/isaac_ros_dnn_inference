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

import errno
import os
import time

from isaac_ros_tensor_msgs.msg import TensorList
from isaac_ros_test import IsaacROSBaseTest
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch_testing

import pytest
import rclpy
from tensor_msgs.msg import ExperimentalTensor


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all TensorRT ROS 2 nodes for testing."""
    # By default loads and runs mobilenetv2-1.0
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_file_path = dir_path + '/models/mobilenetv2-1.0.onnx'

    # Remove the default trt_engine.plan before starting the node if it exists
    try:
        os.remove('/tmp/trt_engine.plan')
        print('Deleted existing /tmp/trt_engine.plan')
    except OSError as e:
        if e.errno != errno.ENOENT:
            print('File exists but error deleting /tmp/trt_engine.plan ')

    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        namespace=IsaacROSTensorRTNodeTest.generate_namespace(),
        parameters=[{
            'model_file_path': model_file_path,
            'output_binding_names': ['mobilenetv20_output_flatten0_reshape0'],
            'output_tensor_names': ['output'],
            'input_tensor_names': ['input'],
            'input_binding_names': ['data'],
            'verbose': False
        }]
    )

    return IsaacROSTensorRTNodeTest.generate_test_description([
        ComposableNodeContainer(
            name='tensor_rt_container',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[tensor_rt_node],
            namespace=IsaacROSTensorRTNodeTest.generate_namespace(),
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        ),
        launch.actions.TimerAction(
            period=5.0, actions=[launch_testing.actions.ReadyToTest()])
    ])


class IsaacROSTensorRTNodeTest(IsaacROSBaseTest):
    """
    Proof-of-Life Test for Isaac ROS TensorRT Node.

    1. Sets up TensorRTNode and TestTensorPublisherNode to send zero tensors to it
    2. Sets up ROS subscriber to listen to output channel of TensorRTNode
    3. Verify received tensors are the correct dimensions (based on default mobilenetv2-1.0)
    """

    # Using default ROS-GXF Bridge output tensor channel configured in 'run_trt_inference' exe
    SUBSCRIBER_CHANNEL = 'tensor_sub'
    # The amount of seconds to allow TensorRT node to run before verifying received tensors
    # Will depend on time taken for TensorRT engine generation
    PYTHON_SUBSCRIBER_WAIT_SEC = 30.0

    # Mobilenetv2-1.0 output tensor properties to verify.
    # DLPack dtype for float32: code=2 (Float), bits=32, lanes=1.
    NAME = 'output'
    DTYPE_CODE = 2
    DTYPE_BITS = 32
    DTYPE_LANES = 1
    DIMENSIONS = [1, 1000]
    DATA_LENGTH = 4000
    MODEL_GENERATION_TIMEOUT_SEC = 400
    GXF_WAIT_SEC = 10
    MODEL_PATH = '/tmp/trt_engine.plan'

    def test_tensor_rt_node(self) -> None:
        self.node._logger.info(
            f'Generating model (timeout={self.MODEL_GENERATION_TIMEOUT_SEC}s)')
        start_time = time.time()
        wait_cycles = 1
        while not os.path.isfile(self.MODEL_PATH):
            time_diff = time.time() - start_time
            if time_diff > self.MODEL_GENERATION_TIMEOUT_SEC:
                self.fail('Model generation timed out')
            if time_diff > wait_cycles*10:
                self.node._logger.info(
                    f'Waiting for model generation to finish... ({int(time_diff)}s passed)')
                wait_cycles += 1
            time.sleep(1)

        self.node._logger.info(
            f'Model generation was finished (took {(time.time() - start_time)}s)')
        self.node._logger.info(
            f'Waiting {self.GXF_WAIT_SEC}s for the engine to be initialized')

        # Wait for TensorRT Engine to be initialized
        time.sleep(self.GXF_WAIT_SEC)

        self.node._logger.info('Starting Isaac ROS TensorRT Node POL Test')

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
            pub_tensor = ExperimentalTensor()

            pub_tensor.dtype_code = self.DTYPE_CODE
            pub_tensor.dtype_bits = self.DTYPE_BITS
            pub_tensor.dtype_lanes = self.DTYPE_LANES
            pub_tensor.shape = [1, 3, 224, 224]
            pub_tensor.strides = []
            pub_tensor.byte_offset = 0
            pub_tensor.data = [0] * 150528 * 4
            pub_tensor_list.names = ['input']
            pub_tensor_list.tensors = [pub_tensor]

            self.node._logger.info(
                f'Publishing test tensors for {self.PYTHON_SUBSCRIBER_WAIT_SEC}s')
            end_time = time.time() + self.PYTHON_SUBSCRIBER_WAIT_SEC
            while time.time() < end_time:
                tensor_pub.publish(pub_tensor_list)
                rclpy.spin_once(self.node, timeout_sec=0.1)

            # Verify received tensors and log total number of tensors received
            num_tensors_received = len(received_messages[subscriber_topic_namespace])
            self.assertGreater(num_tensors_received, 0)
            self.node._logger.info(
                f'Received {num_tensors_received} tensors in '
                f'{self.PYTHON_SUBSCRIBER_WAIT_SEC} seconds')

            for tensor_list, _ in received_messages[subscriber_topic_namespace]:
                tensor = tensor_list.tensors[0]
                tensor_name = tensor_list.names[0]

                # Verify all tensor properties match that of default mobilenetv2-1.0
                self.assertEqual(tensor_name, self.NAME)
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

            # Log properties of last received tensor
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

            self.node._logger.info('Finished Isaac ROS TensorRT Node POL Test')
        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.assertTrue(self.node.destroy_publisher(tensor_pub))
