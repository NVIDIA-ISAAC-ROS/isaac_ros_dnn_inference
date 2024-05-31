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

    keep_aspect_ratio = LaunchConfiguration('keep_aspect_ratio')
    crop_mode = LaunchConfiguration('crop_mode')
    encoding_desired = LaunchConfiguration('encoding_desired')
    final_tensor_name = LaunchConfiguration('final_tensor_name')

    image_mean = LaunchConfiguration('image_mean')
    image_stddev = LaunchConfiguration('image_stddev')
    num_blocks = LaunchConfiguration('num_blocks')

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

    resize_factor = 1.0
    if not enable_padding:
        width_scalar = input_image_width / network_image_width
        height_scalar = input_image_height / network_image_height
        if width_scalar != height_scalar:
            resize_factor = min(width_scalar, height_scalar)

    resize_image_width = int(network_image_width * resize_factor)
    resize_image_height = int(network_image_height * resize_factor)

    # If we do not attach to a shared component container we have to create our own container.
    dnn_image_encoder_container = Node(
        name=component_container_name_arg,
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen',
        condition=UnlessCondition(attach_to_shared_component_container_arg),
    )

    load_composable_nodes = LoadComposableNodes(
        target_container=component_container_name_arg,
        composable_node_descriptions=[
            ComposableNode(
                name='resize_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[
                    {
                        'output_width': resize_image_width,
                        'output_height': resize_image_height,
                        'num_blocks': num_blocks,
                        'keep_aspect_ratio': keep_aspect_ratio,
                        'encoding_desired': '',
                    }
                ],
                remappings=[
                    ('image', image_input_topic),
                    ('camera_info', camera_info_input_topic),
                ],
            ),
            ComposableNode(
                name='crop_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::CropNode',
                parameters=[
                    {
                        'input_width': resize_image_width,
                        'input_height': resize_image_height,
                        'crop_width': network_image_width,
                        'crop_height': network_image_height,
                        'num_blocks': num_blocks,
                        'crop_mode': crop_mode,
                        'roi_top_left_x': int((resize_image_width - network_image_width) / 2.0),
                        'roi_top_left_y': int((resize_image_height - network_image_height) / 2.0),
                    }
                ],
                remappings=[
                    ('image', 'resize/image'),
                    ('camera_info', 'resize/camera_info'),
                ],
            ),
            ComposableNode(
                name='image_format_converter_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
                parameters=[
                    {
                        'image_width': network_image_width,
                        'image_height': network_image_height,
                        'encoding_desired': encoding_desired,
                    }
                ],
                remappings=[
                    ('image_raw', 'crop/image'),
                    ('image', 'converted/image'),
                ],
            ),
            ComposableNode(
                name='image_to_tensor',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
                parameters=[
                    {
                        'scale': True,
                        'tensor_name': 'image',
                    }
                ],
                remappings=[
                    ('image', 'converted/image'),
                    ('tensor', 'image_tensor'),
                ],
            ),
            ComposableNode(
                name='normalize_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
                parameters=[
                    {
                        'mean': image_mean,
                        'stddev': image_stddev,
                        'input_tensor_name': 'image',
                        'output_tensor_name': 'image'
                    }
                ],
                remappings=[
                    ('tensor', 'image_tensor'),
                ],
            ),
            ComposableNode(
                name='interleaved_to_planar_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
                parameters=[
                    {
                        'input_tensor_shape': [network_image_height, network_image_width, 3],
                        'num_blocks': num_blocks,
                    }
                ],
                remappings=[
                    ('interleaved_tensor', 'normalized_tensor'),
                ],
            ),
            ComposableNode(
                name='reshape_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
                parameters=[
                    {
                        'output_tensor_name': final_tensor_name,
                        'input_tensor_shape': [3, network_image_height, network_image_width],
                        'output_tensor_shape': [1, 3, network_image_height, network_image_width],
                        'num_blocks': num_blocks,
                    }
                ],
                remappings=[
                    ('tensor', 'planar_tensor'),
                    ('reshaped_tensor', tensor_output_topic),
                ],
            ),
        ],
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
            'encoding_desired',
            default_value='rgb8',
            description='The desired image format encoding',
        ),
        DeclareLaunchArgument(
            'final_tensor_name',
            default_value='input_tensor',
            description='The tensor name of the output of image encoder',
        ),
        DeclareLaunchArgument(
            'dnn_image_encoder_namespace',
            default_value='dnn_image_encoder',
            description='The namespace to put the DNN image encoder under',
        ),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
