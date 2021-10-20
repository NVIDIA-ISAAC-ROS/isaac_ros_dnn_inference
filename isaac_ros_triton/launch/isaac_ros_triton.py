# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Load and launch mobilenetv2-1.0 onnx model through Triton node."""
    launch_dir_path = os.path.dirname(os.path.realpath(__file__))
    model_dir_path = launch_dir_path + '/../../test/models'

    triton_node = ComposableNode(
        name='triton',
        package='isaac_ros_triton',
        plugin='isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': 'mobilenetv2-1.0_triton_onnx',
            'model_repository_paths': [model_dir_path],
            'max_batch_size': 0,
            'input_binding_names': ['data'],
            'output_binding_names': ['mobilenetv20_output_flatten0_reshape0'],
            'input_tensor_names': ['input'],
            'output_tensor_names': ['output']
        }]
    )

    triton_container = ComposableNodeContainer(
        name='triton_container',
        namespace='triton',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[triton_node],
        output='screen'
    )

    return launch.LaunchDescription([triton_container])
