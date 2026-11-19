# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""
Run affine.pte (y = 2 * x + 1, CPU only) through ExecuTorchNode and check its replies.

Plain launch_testing, because isaac_ros_test imports cv2 and cv_bridge, which this package
does not need.
"""

from array import array
import os
import time
import unittest

from isaac_ros_tensor_msgs.msg import TensorList
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch_testing
import pytest
import rclpy
from tensor_msgs.msg import ExperimentalTensor

# Its own namespace, so that other ROS nodes on the machine cannot answer or listen.
NAMESPACE = f'/isaac_ros_executorch_test_{os.getpid()}'
SHAPE = [2, 3]
# DLPack data type code for floating point.
FLOAT = 2
REPLY_TIMEOUT_SEC = 60.0
NO_REPLY_WAIT_SEC = 3.0


@pytest.mark.rostest
def generate_test_description():
    model = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'affine.pte')
    return launch.LaunchDescription([
        ComposableNodeContainer(
            name='executorch_test_container',
            namespace=NAMESPACE,
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='isaac_ros_executorch',
                    plugin='nvidia::isaac_ros::dnn_inference::ExecuTorchNode',
                    name='executorch_node',
                    namespace=NAMESPACE,
                    parameters=[{
                        'model_file_path': model,
                        'input_tensor_names': ['input'],
                        'output_tensor_names': ['output'],
                    }],
                ),
            ],
            output='screen',
        ),
        launch_testing.actions.ReadyToTest(),
    ])


def make_request(frame_id, values, name='input'):
    request = TensorList(names=[name], tensors=[ExperimentalTensor(
        shape=SHAPE, dtype_code=FLOAT, dtype_bits=32, dtype_lanes=1,
        data=array('f', values).tobytes())])
    request.header.frame_id = frame_id
    return request


class ExecuTorchNodeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('executorch_node_test', namespace=NAMESPACE)
        cls.replies = []
        cls.node.create_subscription(TensorList, 'tensor_sub', cls.replies.append, 10)
        cls.publisher = cls.node.create_publisher(TensorList, 'tensor_pub', 10)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def reply_to(self, frame_id):
        return next((r for r in self.replies if r.header.frame_id == frame_id), None)

    def check_values(self, frame_id, x):
        # The node may still be loading the program, so keep sending until it answers.
        end = time.time() + REPLY_TIMEOUT_SEC
        while self.reply_to(frame_id) is None and time.time() < end:
            self.publisher.publish(make_request(frame_id, x))
            rclpy.spin_once(self.node, timeout_sec=0.5)
        reply = self.reply_to(frame_id)
        self.assertIsNotNone(reply, f'no reply to {frame_id} within {REPLY_TIMEOUT_SEC} s')
        self.assertEqual(list(reply.names), ['output'])
        tensor = reply.tensors[0]
        self.assertEqual(list(tensor.shape), SHAPE)
        self.assertEqual((tensor.dtype_code, tensor.dtype_bits), (FLOAT, 32))
        y = array('f')
        y.frombytes(bytes(tensor.data))
        self.assertEqual(list(y), [2 * v + 1 for v in x])

    def expect_no_reply(self, request, what):
        self.publisher.publish(request)
        end = time.time() + NO_REPLY_WAIT_SEC
        while time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertIsNone(self.reply_to(request.header.frame_id), f'the node answered {what}')

    def test_replies_and_skips_bad_messages(self):
        # Each bad message is sent on its own: the node keeps one queued message, so a
        # second one sent at once could replace it before the node sees it.
        self.check_values('before', [i - 2.5 for i in range(6)])
        self.expect_no_reply(make_request('empty', []), 'a message without data')
        self.check_values('after_empty', [0.5 * i for i in range(6)])
        self.expect_no_reply(
            make_request('wrong_name', [0.0] * 6, name='other'), 'a message without its input')
        self.check_values('after_wrong_name', [-1.0 * i for i in range(6)])
