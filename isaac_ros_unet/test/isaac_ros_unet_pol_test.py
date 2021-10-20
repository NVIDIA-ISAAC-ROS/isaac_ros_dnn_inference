# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Proof-Of-Life test for the Isaac ROS U-Net package.

    1. Sets up DnnImageEncoderNode, TensorRTNode, UNetDecoderNode
    2. Loads a sample image and publishes it
    3. Subscribes to the relevant topics, waiting for an output from UNetDecodeNode
    4. Verifies that the received output sizes and encodings are correct (based on dummy model)

    Note: the data is not verified because the model is initialized with random weights
"""
import errno
import os
import pathlib
import time

from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch_ros.actions.composable_node_container import ComposableNodeContainer
from launch_ros.descriptions.composable_node import ComposableNode
import numpy as np

import pytest
import rclpy

from sensor_msgs.msg import Image

_TEST_CASE_NAMESPACE = 'unet_node_test'


def generate_random_color_palette(num_classes):
    np.random.seed(0)
    L = []
    for i in range(num_classes):
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        color = r << 16 | g << 8 | b
        L.append(color)
    return L


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description for testing relevant nodes."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_file_path = dir_path + '/dummy_model/model.dummy.onnx'
    try:
        os.remove('/tmp/trt_engine.plan')
        print('Deleted existing /tmp/trt_engine.plan')
    except OSError as e:
        if e.errno != errno.ENOENT:
            print('File exists but error deleting /tmp/trt_engine.plan')

    encoder_node = ComposableNode(
        package='isaac_ros_dnn_encoders',
        plugin='isaac_ros::dnn_inference::DnnImageEncoderNode',
        namespace=IsaacROSUNetPipelineTest.generate_namespace(_TEST_CASE_NAMESPACE),
        parameters=[{
            'network_image_width': 960,
            'network_image_height': 544,
            'network_image_encoding': 'rgb8',
            'network_normalization_type': 'positive_negative',
            'tensor_name': 'input_tensor'
        }],
        remappings=[('encoded_tensor', 'tensor_pub')]
    )

    tensorrt_node = ComposableNode(
        package='isaac_ros_tensor_rt',
        plugin='isaac_ros::dnn_inference::TensorRTNode',
        namespace=IsaacROSUNetPipelineTest.generate_namespace(_TEST_CASE_NAMESPACE),
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': '/tmp/trt_engine.plan',
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input_1'],
            'output_tensor_names': ['output_tensor'],
            'output_binding_names': ['softmax_1'],
            'verbose': False
        }])

    unet_decoder_node = ComposableNode(
        package='isaac_ros_unet',
        plugin='isaac_ros::unet::UNetDecoderNode',
        namespace=IsaacROSUNetPipelineTest.generate_namespace(_TEST_CASE_NAMESPACE),
        parameters=[{
            'frame_id': 'unet',
            'color_segmentation_mask_encoding': 'rgb8',
            'color_palette': generate_random_color_palette(20),  # 20 classes
        }])

    container = ComposableNodeContainer(
        name='unet_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[encoder_node, tensorrt_node, unet_decoder_node],
        output='screen'
    )

    return IsaacROSUNetPipelineTest.generate_test_description([container])


class IsaacROSUNetPipelineTest(IsaacROSBaseTest):
    """Validates a U-Net model with randomized weights with a sample output from Python."""

    filepath = pathlib.Path(os.path.dirname(__file__))
    MODEL_GENERATION_TIMEOUT_SEC = 300
    INIT_WAIT_SEC = 10
    MODEL_PATH = '/tmp/trt_engine.plan'

    @IsaacROSBaseTest.for_each_test_case()
    def test_image_segmentation(self, test_folder):
        start_time = time.time()
        while not os.path.isfile(self.MODEL_PATH):
            if (time.time() - start_time) > self.MODEL_GENERATION_TIMEOUT_SEC:
                self.fail('Model generation timed out')
            time.sleep(1)
        # Wait for TensorRT engine
        time.sleep(self.INIT_WAIT_SEC)

        """Expect the node to segment an image."""
        self.generate_namespace_lookup(
            ['image', 'unet/raw_segmentation_mask',
             'unet/colored_segmentation_mask'], _TEST_CASE_NAMESPACE)
        image_pub = self.node.create_publisher(Image, self.namespaces['image'], self.DEFAULT_QOS)
        received_messages = {}
        segmentation_mask_sub, color_segmentation_mask_sub = self.create_logging_subscribers(
            [('unet/raw_segmentation_mask', Image),
             ('unet/colored_segmentation_mask', Image)
             ], received_messages, accept_multiple_messages=False)

        EXPECTED_HEIGHT = 544
        EXPECTED_WIDTH = 960

        try:
            image = JSONConversion.load_image_from_json(test_folder / 'image.json')
            TIMEOUT = 60
            end_time = time.time() + TIMEOUT
            done = False
            while time.time() < end_time:
                image_pub.publish(image)
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'unet/raw_segmentation_mask' in received_messages and \
                   'unet/colored_segmentation_mask' in received_messages:
                    done = True
                    break

            self.assertTrue(done, "Didn't receive output on unet/raw_segmentation_mask topic!")

            unet_mask = received_messages['unet/raw_segmentation_mask']

            self.assertEqual(unet_mask.width, EXPECTED_WIDTH, 'Received incorrect width')
            self.assertEqual(unet_mask.height, EXPECTED_HEIGHT, 'Received incorrect height')
            self.assertEqual(unet_mask.encoding, 'mono8', 'Received incorrect encoding')

            unet_color_mask = received_messages['unet/colored_segmentation_mask']
            self.assertEqual(unet_color_mask.width, EXPECTED_WIDTH,
                             'Received incorrect width for colored mask!')
            self.assertEqual(unet_color_mask.height, EXPECTED_HEIGHT,
                             'Received incorrect height for colored mask!')
            self.assertEqual(unet_color_mask.encoding, 'rgb8', 'Received incorrect encoding!')
        finally:
            self.assertTrue(self.node.destroy_subscription(segmentation_mask_sub))
            self.assertTrue(self.node.destroy_subscription(color_segmentation_mask_sub))
            self.assertTrue(self.node.destroy_publisher(image_pub))
