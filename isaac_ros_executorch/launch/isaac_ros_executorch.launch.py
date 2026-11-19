# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Run the ExecuTorch inference node. The tensor names default to the ones Isaac ROS DOPE uses."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def launch_setup(context):
    def as_list(name):
        value = LaunchConfiguration(name).perform(context)
        return [item.strip() for item in value.split(',')] if value else []

    parameters = {
        'model_file_path': LaunchConfiguration('model_file_path').perform(context),
        'input_tensor_names': as_list('input_tensor_names'),
        'output_tensor_names': as_list('output_tensor_names'),
    }
    # A list parameter cannot be empty in a launch file, so leave it out to get the node's default.
    data_files = as_list('data_file_paths')
    if data_files:
        parameters['data_file_paths'] = data_files
    return [
        ComposableNodeContainer(
            name='executorch_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                ComposableNode(
                    package='isaac_ros_executorch',
                    plugin='nvidia::isaac_ros::dnn_inference::ExecuTorchNode',
                    name='executorch_node',
                    parameters=[parameters],
                ),
            ],
            output='screen',
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_file_path', description='Path to the .pte file'),
        DeclareLaunchArgument(
            'data_file_paths', default_value='',
            description='Comma separated .ptd files; pass every one the export wrote'),
        DeclareLaunchArgument(
            'input_tensor_names', default_value='input_tensor',
            description='Comma separated input tensor names, not a YAML list '
                        '(the DOPE names by default)'),
        DeclareLaunchArgument(
            'output_tensor_names', default_value='output',
            description='Comma separated output tensor names, not a YAML list '
                        '(the DOPE names by default)'),
        OpaqueFunction(function=launch_setup),
    ])
