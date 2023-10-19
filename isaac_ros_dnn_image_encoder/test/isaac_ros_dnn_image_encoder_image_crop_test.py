# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from math import ceil
import os
import pathlib
import struct
import time

from cv_bridge import CvBridge
from isaac_ros_tensor_list_interfaces.msg import TensorList
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import numpy as np

import pytest
import rclpy

from sensor_msgs.msg import Image


INPUT_IMAGE_WIDTH = 1920
INPUT_IMAGE_HEIGHT = 1080

NETWORK_IMAGE_WIDTH = 512
NETWORK_IMAGE_HEIGHT = 512
IMAGE_MEAN = np.array([0.5, 0.5, 0.5])
IMAGE_STDDEV = np.array([0.5, 0.5, 0.5])


@pytest.mark.rostest
def generate_test_description():
    encoder_node = ComposableNode(
        name='encoder',
        package='isaac_ros_dnn_image_encoder',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        namespace=IsaacROSDnnImageEncoderImageResizeNodeTest.generate_namespace(),
        parameters=[{
            'input_image_width': INPUT_IMAGE_WIDTH,
            'input_image_height': INPUT_IMAGE_HEIGHT,
            'network_image_width': NETWORK_IMAGE_WIDTH,
            'network_image_height': NETWORK_IMAGE_HEIGHT,
            'image_mean': list(IMAGE_MEAN),
            'image_stddev': list(IMAGE_STDDEV),
            'enable_padding': True
        }],
        remappings=[('encoded_tensor', 'tensors')])

    return IsaacROSDnnImageEncoderImageResizeNodeTest.generate_test_description([
        ComposableNodeContainer(
            name='tensor_rt_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[encoder_node],
            namespace=IsaacROSDnnImageEncoderImageResizeNodeTest.generate_namespace(),
            output='screen',
            arguments=['--ros-args', '--log-level', 'info',
                       '--log-level', 'isaac_ros_test.encoder:=debug'],
        )
    ])


class IsaacROSDnnImageEncoderImageResizeNodeTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_image_resize(self):
        """Test Image Resize feature."""
        TIMEOUT = 300
        received_messages = {}

        self.generate_namespace_lookup(['image', 'tensors'])

        image_pub = self.node.create_publisher(
            Image, self.namespaces['image'], self.DEFAULT_QOS)

        subs = self.create_logging_subscribers(
            [('tensors', TensorList)], received_messages, accept_multiple_messages=False)

        try:
            # Create image with colored pixels
            cv_image = np.ones((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 3), np.uint8) * 255

            # What fraction of each dimension should be colored for tracing
            TRACER_PATCH_SIZE_FRACTION = 0.5

            # Patch guaranteed to be at least 1 pixel large
            TRACER_PATCH_HEIGHT = ceil(TRACER_PATCH_SIZE_FRACTION * INPUT_IMAGE_HEIGHT)
            TRACER_PATCH_WIDTH = ceil(TRACER_PATCH_SIZE_FRACTION * INPUT_IMAGE_WIDTH)

            # Input image layout:
            # --------------------
            # | R R R R  G G G G |
            # | R R R R  G G G G |
            # | R R R R  G G G G |
            # |                  |
            # | B B B B  W W W W |
            # | B B B B  W W W W |
            # | B B B B  W W W W |
            # -------------------

            # Red pixels in top left corner
            cv_image[:TRACER_PATCH_HEIGHT, :TRACER_PATCH_WIDTH] = (0, 0, 255)

            # Green pixels in top right corner
            cv_image[:TRACER_PATCH_HEIGHT, -TRACER_PATCH_WIDTH:] = (0, 255, 0)

            # Blue pixels in bottom left corner
            cv_image[-TRACER_PATCH_HEIGHT:, :TRACER_PATCH_WIDTH] = (255, 0, 0)

            image = CvBridge().cv2_to_imgmsg(cv_image)
            image.encoding = 'bgr8'

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

            SIZEOF_FLOAT = 4
            self.assertTrue(
                len(tensor.data) / SIZEOF_FLOAT ==
                NETWORK_IMAGE_HEIGHT * NETWORK_IMAGE_WIDTH * 3,
                'Tensor did not have the expected length!'
            )

            def offset_nchw(n, c, h, w):
                # Tensor has been encoded in NCHW format
                # N = 1                         # Since only one image has been sent, N = 1
                C = 3                           # Tensor encoding is R, G, B ordering
                H = NETWORK_IMAGE_HEIGHT    # Output height
                W = NETWORK_IMAGE_WIDTH     # Output width
                return n * C * H * W + c * H * W + h * W + w

            def extract_pixel(data, x, y):
                return (
                    # Convert bytes to float representing color channel
                    struct.unpack('<f', data[
                        SIZEOF_FLOAT * offset:
                        SIZEOF_FLOAT * (offset + 1)
                    ])[0]  # struct.unpack returns a tuple with one element
                    for offset in (
                        # Calculate byte offsets for each color channel
                        offset_nchw(0, i, y, x) for i in range(3)
                    )
                )

            # Compute expected values corresponding to R, G, B after normalization
            RED_EXPECTED_VAL, GREEN_EXPECTED_VAL, BLUE_EXPECTED_VAL = (
                                                                    1 - IMAGE_MEAN) / IMAGE_STDDEV
            COLOR_MATCH_TOLERANCE = 0.05

            red_pixel_count = 0
            green_pixel_count = 0
            blue_pixel_count = 0
            white_pixel_count = 0
            black_pixel_count = 0

            # Expected o/p tensor layout:
            # --------------------
            # | Black Black Black|
            # | R R R R  G G G G |
            # | R R R R  G G G G |
            # | R R R R  G G G G |
            # | R R R R  G G G G |
            # |                  |
            # | B B B B  W W W W |
            # | B B B B  W W W W |
            # | B B B B  W W W W |
            # | B B B B  W W W W |
            # | Black Black Black|
            # --------------------

            EXPECTED_NUM_RED_PIXELS = (NETWORK_IMAGE_WIDTH *
                                       (float(INPUT_IMAGE_HEIGHT) / float(INPUT_IMAGE_WIDTH))) *\
                TRACER_PATCH_SIZE_FRACTION * (NETWORK_IMAGE_WIDTH / 2)

            EXPECTED_NUM_GREEN_PIXELS = EXPECTED_NUM_RED_PIXELS
            EXPECTED_NUM_BLUE_PIXELS = EXPECTED_NUM_RED_PIXELS
            EXPECTED_NUM_WHITE_PIXELS = EXPECTED_NUM_RED_PIXELS

            EXPECTED_NUM_BLACK_PIXELS = (NETWORK_IMAGE_WIDTH - NETWORK_IMAGE_WIDTH *
                                         (float(INPUT_IMAGE_HEIGHT) / float(INPUT_IMAGE_WIDTH))) *\
                TRACER_PATCH_SIZE_FRACTION * (NETWORK_IMAGE_WIDTH) * 2

            for y in range(NETWORK_IMAGE_HEIGHT):
                for x in range(NETWORK_IMAGE_WIDTH):
                    # Extract 3 float values corresponding to the
                    r, g, b = extract_pixel(tensor.data, x, y)

                    if(abs(r - RED_EXPECTED_VAL) < COLOR_MATCH_TOLERANCE
                       and g < COLOR_MATCH_TOLERANCE and b < COLOR_MATCH_TOLERANCE):
                        red_pixel_count += 1
                    if(abs(g - GREEN_EXPECTED_VAL) < COLOR_MATCH_TOLERANCE
                       and r < COLOR_MATCH_TOLERANCE and b < COLOR_MATCH_TOLERANCE):
                        green_pixel_count += 1
                    if(abs(b - BLUE_EXPECTED_VAL) < COLOR_MATCH_TOLERANCE
                       and r < COLOR_MATCH_TOLERANCE and g < COLOR_MATCH_TOLERANCE):
                        blue_pixel_count += 1
                    if(abs(r - RED_EXPECTED_VAL) < COLOR_MATCH_TOLERANCE and
                       abs(g - GREEN_EXPECTED_VAL) < COLOR_MATCH_TOLERANCE and
                       abs(b - BLUE_EXPECTED_VAL) < COLOR_MATCH_TOLERANCE):
                        white_pixel_count += 1
                    if(r < COLOR_MATCH_TOLERANCE and g < COLOR_MATCH_TOLERANCE and
                       b < COLOR_MATCH_TOLERANCE):
                        black_pixel_count += 1

            self.assertEqual(red_pixel_count, EXPECTED_NUM_RED_PIXELS,
                             msg='Count of red pixles do not match')
            self.assertEqual(green_pixel_count, EXPECTED_NUM_GREEN_PIXELS,
                             msg='Count of green pixles do not match')
            self.assertEqual(blue_pixel_count, EXPECTED_NUM_BLUE_PIXELS,
                             msg='Count of blue pixles do not match')
            self.assertEqual(white_pixel_count, EXPECTED_NUM_WHITE_PIXELS,
                             msg='Count of white pixles do not match')
            self.assertEqual(black_pixel_count, EXPECTED_NUM_BLACK_PIXELS,
                             msg='Count of black pixles do not match')

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
