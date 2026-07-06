# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from ament_index_python.packages import get_package_share_directory
from isaac_ros_tensor_list_interfaces.msg import TensorList
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description that runs dnn_image_encoder_nodes.launch.py."""
    namespace = IsaacROSDnnImageEncoderSeparateNodesTest.generate_namespace()

    separate_nodes_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('isaac_ros_dnn_image_encoder'),
                'launch',
                'dnn_image_encoder_nodes.launch.py',
            )
        ),
        launch_arguments={
            'input_image_width': '1920',
            'input_image_height': '1080',
            'network_image_width': '512',
            'network_image_height': '512',
            'dnn_image_encoder_namespace': namespace,
            'tensor_output_topic': 'tensors',
            'image_input_topic': 'image',
            'camera_info_input_topic': 'camera_info',
        }.items(),
    )

    return IsaacROSDnnImageEncoderSeparateNodesTest.generate_test_description([
        separate_nodes_launch,
    ])


class IsaacROSDnnImageEncoderSeparateNodesTest(IsaacROSBaseTest):
    """Test the dnn_image_encoder pipeline from dnn_image_encoder_nodes.launch.py."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_separate_nodes_pipeline(self):
        """
        Test the separate-nodes image encoder pipeline.

        Launches resize -> format converter -> crop -> image_to_tensor -> normalize
        -> interleaved_to_planar -> reshape and verifies that publishing an image
        and camera_info produces a correctly shaped NCHW tensor (1, 3, 512, 512).
        """
        TIMEOUT = 300
        received_messages = {}

        self.generate_namespace_lookup(['image', 'camera_info', 'tensors'])

        image_pub = self.node.create_publisher(
            Image, self.namespaces['image'], self.DEFAULT_QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS)

        subs = self.create_logging_subscribers(
            [('tensors', TensorList)], received_messages)

        try:
            json_file = self.filepath / 'test_cases/pose_estimation_0/image.json'
            image = JSONConversion.load_image_from_json(json_file)
            info_file = self.filepath / 'test_cases/pose_estimation_0/camera_info.json'
            camera_info = JSONConversion.load_camera_info_from_json(info_file)
            timestamp = self.node.get_clock().now().to_msg()
            image.header.stamp = timestamp
            camera_info.header = image.header

            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                image_pub.publish(image)
                camera_info_pub.publish(camera_info)
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if 'tensors' in received_messages:
                    done = True
                    break

            self.assertTrue(done, 'Appropriate output not received')
            tensor_list = received_messages['tensors']
            tensor = tensor_list.tensors[0]

            # Pipeline may not propagate header stamp through all nodes; verify shape only
            # Expect NCHW shape [1, 3, 512, 512]
            self.assertEqual(tensor.shape.rank, 4,
                             'Expected rank 4 tensor')
            self.assertEqual(tensor.shape.dims[0], 1, 'Expected batch dim 1')
            self.assertEqual(tensor.shape.dims[1], 3, 'Expected channel dim 3')
            self.assertEqual(tensor.shape.dims[2], 512, 'Expected height 512')
            self.assertEqual(tensor.shape.dims[3], 512, 'Expected width 512')

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
