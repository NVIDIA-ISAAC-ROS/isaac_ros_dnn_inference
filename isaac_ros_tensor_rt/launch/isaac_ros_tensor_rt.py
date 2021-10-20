# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for TensorRT ROS2 node."""
    # By default loads and runs mobilenetv2-1.0 included in isaac_ros_dnn_inference/models
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_file_path = dir_path + '/../../test/models/mobilenetv2-1.0.onnx'

    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='isaac_ros::dnn_inference::TensorRTNode',
        remappings=[('tensor_pub', 'tensor_pub'),
                    ('tensor_sub', 'tensor_sub')],
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': '/tmp/trt_engine.plan',
            'output_binding_names': ['mobilenetv20_output_flatten0_reshape0'],
            'output_tensor_names': ['output'],
            'input_tensor_names': ['input'],
            'input_binding_names': ['data'],
            'verbose': False,
            'force_engine_update': True
        }]
    )

    return LaunchDescription([
        ComposableNodeContainer(
            name='tensor_rt_container',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[tensor_rt_node],
            namespace='isaac_ros_tensor_rt'
        )
    ])
