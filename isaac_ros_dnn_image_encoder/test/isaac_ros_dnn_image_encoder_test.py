# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cv2
from cv_bridge import CvBridge
from isaac_ros_tensor_list_interfaces.msg import TensorList
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image


@pytest.mark.rostest
def generate_test_description():
    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    encoder_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': '1920',
            'input_image_height': '1080',
            'network_image_width': '512',
            'network_image_height': '512',
            'enable_padding': 'True',
            'tensor_output_topic': 'tensors',
            'dnn_image_encoder_namespace': IsaacROSDnnImageEncoderNodeTest.generate_namespace(),
        }.items(),
    )

    return IsaacROSDnnImageEncoderNodeTest.generate_test_description([
        encoder_node_launch,
    ])


class IsaacROSDnnImageEncoderNodeTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_maintain_aspect_ratio(self):
        """
        Test maintain aspect ratio feature.

        Test that the DNN Image encoder is cropping and resizing to maintain aspect
        ratio properly. The test will crop the image to 1080 by 1080 , resize it to
        512 by 512, and then converting it to format NCHW.
        """
        TIMEOUT = 300
        received_messages = {}

        self.generate_namespace_lookup(['image', 'camera_info', 'tensors'])

        image_pub = self.node.create_publisher(
            Image, self.namespaces['image'], self.DEFAULT_QOS)

        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS)

        # The current DOPE decoder outputs TensorList
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
                rclpy.spin_once(self.node, timeout_sec=(0.1))
                if 'tensors' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Appropriate output not received')
            tensor_list = received_messages['tensors']
            tensor = tensor_list.tensors[0]

            self.assertEqual(str(timestamp), str(tensor_list.header.stamp),
                             'Timestamps do not match.')

            cv_image = CvBridge().imgmsg_to_cv2(image, desired_encoding='rgb8')

            cv_image = cv_image[:, 0:1080]
            cv_image = cv2.resize(cv_image, (512, 512))
            cv_image = cv2.dnn.blobFromImage(cv_image)

            # Checking the number of dims in tensor. It should be 3 for the input image(RGB)
            self.assertTrue(tensor.shape.rank == 4)
            # Checking the height of resized tensor
            self.assertTrue(tensor.shape.dims[2] == 512)
            # Checking the width of resized tensor
            self.assertTrue(tensor.shape.dims[3] == 512)

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
