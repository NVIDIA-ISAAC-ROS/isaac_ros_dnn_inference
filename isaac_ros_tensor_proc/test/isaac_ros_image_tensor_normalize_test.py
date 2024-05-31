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


DIMENSION_WIDTH = 100
DIMENSION_HEIGHT = 100
DIMENSION_CHANNELS = 3


@pytest.mark.rostest
def generate_test_description():
    normalize_node = ComposableNode(
        name='normalize_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
        namespace=IsaacROSNormalizeNodeTest.generate_namespace(),
        parameters=[{
            'mean': [0.5, 0.6, 0.25],
            'stddev': [0.25, 0.8, 0.5],
        }],
        remappings=[('normalized_tensor', 'tensors')])

    return IsaacROSNormalizeNodeTest.generate_test_description([
        ComposableNodeContainer(
            name='normalize_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[normalize_node],
            namespace=IsaacROSNormalizeNodeTest.generate_namespace(),
            output='screen',
            arguments=['--ros-args', '--log-level', 'info',
                       '--log-level', 'isaac_ros_test.encoder:=debug'],
        )
    ])


class IsaacROSNormalizeNodeTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_image_normalization(self):
        """
        Test Image Normalization feature.

        Test that the NormalizeNode is correctly normalizing the image based on
        the given image mean and standard deviation vectors.
        Given that the image mean vector is <0.5, 0.6, 0.25>, and the image standard
        deviation vector is <0.25, 0.8, 0.5>, and that our input image is white
        (each pixel value is 255), the value for each channel should be:
        RED: ((255 / 255) - 0.5) / 0.25 = 2.0
        GREEN: ((255 / 255) - 0.6) / 0.8 = 0.5
        BLUE: ((255/ 255) - 0.25) / 0.5 = 1.5
        This test verifies that each channel's values should be the calculated values
        above.
        """
        TIMEOUT = 300
        received_messages = {}
        RED_EXPECTED_VAL = 2.0
        GREEN_EXPECTED_VAL = 0.5
        BLUE_EXPECTED_VAL = 1.5
        EXPECTED_VALS = [RED_EXPECTED_VAL, GREEN_EXPECTED_VAL,
                         BLUE_EXPECTED_VAL]

        TENSOR_NAME = 'tensor'
        TENSOR_DATA_TYPE = 9
        TENSOR_DIMENSIONS = [DIMENSION_HEIGHT, DIMENSION_WIDTH, DIMENSION_CHANNELS]
        TENSOR_RANK = len(TENSOR_DIMENSIONS)

        self.generate_namespace_lookup(['tensor', 'image', 'tensors'])

        image_pub = self.node.create_publisher(
            TensorList, self.namespaces['tensor'], self.DEFAULT_QOS)

        subs = self.create_logging_subscribers(
            [('tensors', TensorList)], received_messages)

        try:
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
            tensor_data[:] = 1.0
            tensor.data = tensor_data.tobytes()
            tensor_list.tensors = [tensor]

            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                image_pub.publish(tensor_list)
                rclpy.spin_once(self.node, timeout_sec=(0.1))
                if 'tensors' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Appropriate output not received')
            tensor = received_messages['tensors'].tensors[0]

            # The tensor has the format HWC and is a float array, so
            # use numpy to interpret it as such, and then reshape it
            normalized_tensor = np.frombuffer(tensor.data, np.float32)
            normalized_tensor = normalized_tensor.reshape(DIMENSION_HEIGHT,
                                                          DIMENSION_WIDTH,
                                                          DIMENSION_CHANNELS)
            for c in range(DIMENSION_CHANNELS):
                self.assertTrue(
                    (np.round(normalized_tensor[:, :, c], 1) == EXPECTED_VALS[c]).all()
                )
        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
