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

import ast

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, OpaqueFunction)
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes, Node, PushRosNamespace
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    input_image_width = int(
        context.perform_substitution(LaunchConfiguration('input_image_width'))
    )
    input_image_height = int(
        context.perform_substitution(LaunchConfiguration('input_image_height'))
    )

    network_image_width = int(
        context.perform_substitution(LaunchConfiguration('network_image_width'))
    )
    network_image_height = int(
        context.perform_substitution(LaunchConfiguration('network_image_height'))
    )
    enable_padding = ast.literal_eval(
        context.perform_substitution(LaunchConfiguration('enable_padding'))
    )
    tensor_name = LaunchConfiguration('tensor_name', default='output_tensor')

    input_encoding = LaunchConfiguration('input_encoding')

    image_mean = LaunchConfiguration('image_mean')
    image_stddev = LaunchConfiguration('image_stddev')

    image_input_topic = LaunchConfiguration('image_input_topic', default='image')
    camera_info_input_topic = LaunchConfiguration('camera_info_input_topic', default='camera_info')
    tensor_output_topic = LaunchConfiguration('tensor_output_topic', default='encoded_tensor')

    attach_to_shared_component_container_arg = LaunchConfiguration(
        'attach_to_shared_component_container', default=False
    )
    component_container_name_arg = LaunchConfiguration(
        'component_container_name', default='dnn_image_encoder_container'
    )
    dnn_image_encoder_namespace = LaunchConfiguration('dnn_image_encoder_namespace')

    # If we do not attach to a shared component container we have to create our own container.
    dnn_image_encoder_container = Node(
        name=component_container_name_arg,
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen',
        condition=UnlessCondition(attach_to_shared_component_container_arg),
    )

    dnn_image_encoder_node = ComposableNode(
        name='dnn_image_encoder_node',
        package='isaac_ros_dnn_image_encoder',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        namespace=dnn_image_encoder_namespace,
        parameters=[{
            'input_image_width': input_image_width,
            'input_image_height': input_image_height,
            'network_image_width': network_image_width,
            'network_image_height': network_image_height,
            'input_encoding': input_encoding,
            'image_mean': image_mean,
            'image_stddev': image_stddev,
            'enable_padding': enable_padding,
            'tensor_output_topic': tensor_output_topic,
            'dnn_image_encoder_namespace': dnn_image_encoder_namespace,
            'tensor_name': tensor_name,
        }],
        remappings=[
            ('image', image_input_topic),
            ('tensors', tensor_output_topic),
            ('camera_info', camera_info_input_topic),
        ],
    )

    load_composable_nodes = LoadComposableNodes(
        target_container=component_container_name_arg,
        composable_node_descriptions=[dnn_image_encoder_node],
    )

    final_launch = GroupAction(
        actions=[
            dnn_image_encoder_container,
            PushRosNamespace(dnn_image_encoder_namespace),
            load_composable_nodes,
        ],
    )
    return [final_launch]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'input_image_width',
            default_value='0',
            description='The input image width',
        ),
        DeclareLaunchArgument(
            'input_image_height',
            default_value='0',
            description='The input image height',
        ),
        DeclareLaunchArgument(
            'network_image_width',
            default_value='0',
            description='The network image width',
        ),
        DeclareLaunchArgument(
            'network_image_height',
            default_value='0',
            description='The network image height',
        ),
        DeclareLaunchArgument(
            'image_mean',
            default_value='[0.5, 0.5, 0.5]',
            description='The mean for image normalization',
        ),
        DeclareLaunchArgument(
            'image_stddev',
            default_value='[0.5, 0.5, 0.5]',
            description='The standard deviation for image normalization',
        ),
        DeclareLaunchArgument(
            'enable_padding',
            default_value='True',
            description='Whether to enable padding or not',
        ),
        DeclareLaunchArgument(
            'input_qos',
            default_value='DEFAULT',
            description='The QoS settings for the input image'
        ),
        DeclareLaunchArgument(
            'output_qos',
            default_value='DEFAULT',
            description='The QoS settings for the output tensor'
        ),
        DeclareLaunchArgument(
            'num_blocks',
            default_value='40',
            description='The number of preallocated memory blocks',
        ),
        DeclareLaunchArgument(
            'keep_aspect_ratio',
            default_value='True',
            description='Whether to maintain the aspect ratio or not while resizing'
        ),
        DeclareLaunchArgument(
            'crop_mode',
            default_value='CENTER',
            description='The crop mode to crop the image using',
        ),
        DeclareLaunchArgument(
            'input_encoding',
            default_value='rgb8',
            description='The desired image format encoding',
        ),
        DeclareLaunchArgument(
            'dnn_image_encoder_namespace',
            default_value='dnn_image_encoder',
            description='The namespace to put the DNN image encoder under',
        ),
        DeclareLaunchArgument(
            'tensor_name',
            default_value='output_tensor',
            description='The name of the output tensor',
        ),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
