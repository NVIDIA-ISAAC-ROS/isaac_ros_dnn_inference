# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pathlib
import time

from isaac_ros_tensor_list_interfaces.msg import Tensor, TensorList, TensorShape
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import numpy as np

import pytest
import rclpy


DIMENSION_WIDTH = 75
DIMENSION_HEIGHT = 100
DIMENSION_CHANNELS = 3


@pytest.mark.rostest
def generate_test_description():
    reshape_node = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        namespace=IsaacROSDnnReshapeNodeTest.generate_namespace(),
        parameters=[{
            'input_tensor_shape': [DIMENSION_HEIGHT, DIMENSION_WIDTH,
                                   DIMENSION_CHANNELS],
            'output_tensor_shape': [1, DIMENSION_HEIGHT, DIMENSION_WIDTH,
                                    DIMENSION_CHANNELS],
            'output_tensor_name': 'output',
        }],
        remappings=[
            ('reshaped_tensor', 'tensors')
        ])

    return IsaacROSDnnReshapeNodeTest.generate_test_description([
        ComposableNodeContainer(
            name='tensor_rt_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[reshape_node],
            namespace=IsaacROSDnnReshapeNodeTest.generate_namespace(),
            output='screen',
            arguments=['--ros-args', '--log-level', 'info',
                       '--log-level', 'isaac_ros_test.encoder:=debug'],
        )
    ])


class IsaacROSDnnReshapeNodeTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_reshape(self):
        """
        Test Tensor Reshape feature.

        This test simply takes a HWC tensor (100x75x3). The first channel is
        filled with 0, the second channel with 1 and the last channel with 2.
        Afterward, it tests the shape of the new tensor with 1x100x75x3
        """
        TIMEOUT = 300

        TENSOR_NAME = 'input_tensor'
        TENSOR_DATA_TYPE = 9
        TENSOR_DIMENSIONS = [DIMENSION_HEIGHT, DIMENSION_WIDTH, DIMENSION_CHANNELS]
        TENSOR_RANK = len(TENSOR_DIMENSIONS)

        received_messages = {}

        self.generate_namespace_lookup(['tensor', 'tensors'])

        tensor_pub = self.node.create_publisher(
            TensorList, self.namespaces['tensor'], self.DEFAULT_QOS)

        subs = self.create_logging_subscribers(
            [('tensors', TensorList)], received_messages)

        try:
            # Create tensor
            tensor_list = TensorList()
            tensor = Tensor()
            shape = TensorShape()

            shape.rank = TENSOR_RANK
            shape.dims = TENSOR_DIMENSIONS
            tensor.shape = shape

            tensor.name = TENSOR_NAME
            tensor.data_type = TENSOR_DATA_TYPE
            # NOTE: we let NITROS handle stride calculation, etc
            tensor.strides = []
            tensor_data = np.zeros((DIMENSION_HEIGHT, DIMENSION_WIDTH,
                                    DIMENSION_CHANNELS), np.float32)
            tensor_data[:] = list(range(DIMENSION_CHANNELS))
            tensor.data = tensor_data.tobytes()
            tensor_list.tensors = [tensor]

            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                tensor_pub.publish(tensor_list)
                rclpy.spin_once(self.node, timeout_sec=(0.1))
                if 'tensors' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Appropriate output not received')
            result_tensor_list = received_messages['tensors']

            self.assertEqual(len(result_tensor_list.tensors), 1)
            result_tensor = result_tensor_list.tensors[0]
            self.assertEqual(result_tensor.shape.rank, TENSOR_RANK + 1)
            self.assertEqual(result_tensor.name, 'output')

            RESULTANT_DIMS = [1, DIMENSION_HEIGHT, DIMENSION_WIDTH,
                              DIMENSION_CHANNELS]

            self.assertEqual(result_tensor.shape.dims.tolist(),
                             RESULTANT_DIMS)
            self.assertEqual(result_tensor.data_type, TENSOR_DATA_TYPE)

            resultant_data = np.frombuffer(result_tensor.data, np.float32)
            resultant_data = np.reshape(resultant_data, tuple(RESULTANT_DIMS))
            for i in range(DIMENSION_CHANNELS):
                self.assertTrue((resultant_data[0, :, :, i] == i).all())

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(tensor_pub)
