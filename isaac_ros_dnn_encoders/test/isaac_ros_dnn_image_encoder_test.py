# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import pathlib
import struct
import time

import cv2
from cv_bridge import CvBridge
from isaac_ros_nvengine_interfaces.msg import TensorList
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import pytest
import rclpy

from sensor_msgs.msg import Image


@pytest.mark.rostest
def generate_test_description():
    encoder_node = ComposableNode(
        name='encoder',
        package='isaac_ros_dnn_encoders',
        plugin='isaac_ros::dnn_inference::DnnImageEncoderNode',
        namespace=IsaacROSDnnImageEncoderNodeTest.generate_namespace(),
        parameters=[{
            'network_image_width': 512,
            'network_image_height': 512,
            'network_image_encoding': 'rgb8',
            'network_normalization_type': 'none',
            'maintain_aspect_ratio': True
        }],
        remappings=[('encoded_tensor', 'tensors')])

    return IsaacROSDnnImageEncoderNodeTest.generate_test_description([
        ComposableNodeContainer(
            name='tensor_rt_container',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[encoder_node],
            namespace=IsaacROSDnnImageEncoderNodeTest.generate_namespace(),
            output='screen'
        )
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

        self.generate_namespace_lookup(['image', 'tensors'])

        image_pub = self.node.create_publisher(
            Image, self.namespaces['image'], self.DEFAULT_QOS)

        # The current DOPE decoder outputs TensorList
        subs = self.create_logging_subscribers(
            [('tensors', TensorList)], received_messages)

        try:
            json_file = self.filepath / 'test_cases/pose_estimation_0/image.json'
            image = JSONConversion.load_image_from_json(json_file)

            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                image_pub.publish(image)
                rclpy.spin_once(self.node, timeout_sec=(0.1))
                if 'tensors' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Appropriate output not received')
            tensor = received_messages['tensors'].tensors[0]

            cv_image = CvBridge().imgmsg_to_cv2(image, desired_encoding='rgb8')

            cv_image = cv_image[:, 0:1080]
            cv_image = cv2.resize(cv_image, (512, 512))
            cv_image = cv2.dnn.blobFromImage(cv_image)

            flattened_cv_image = cv_image.flatten()
            for i in range(0, len(flattened_cv_image)):
                result_val = struct.unpack('<f', tensor.data[4*i:4*i+4])
                self.assertTrue(flattened_cv_image[i] == result_val)

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
