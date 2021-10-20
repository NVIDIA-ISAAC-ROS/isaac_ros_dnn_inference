# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launch the DNN Image encoder, Triton node and UNet decoder node."""
    # Parameters preconfigured for PeopleSemSegNet.
    encoder_node = ComposableNode(
        name='dnn_image_encoder',
        package='isaac_ros_dnn_encoders',
        plugin='isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            'network_image_width': 960,
            'network_image_height': 544,
            'network_image_encoding': 'rgb8',
            'network_normalization_type': 'positive_negative',
            'tensor_name': 'input_tensor'
        }],
        remappings=[('encoded_tensor', 'tensor_pub')]
    )

    triton_node = ComposableNode(
        name='triton_node',
        package='isaac_ros_triton',
        plugin='isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': '',
            'model_repository_paths': [''],
            'max_batch_size': 0,
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input_1'],
            'output_tensor_names': ['output_tensor'],
            'output_binding_names': ['softmax_1'],
        }])

    unet_decoder_node = ComposableNode(
        name='unet_decoder_node',
        package='isaac_ros_unet',
        plugin='isaac_ros::unet::UNetDecoderNode',
        parameters=[{
            'frame_id': 'unet',
            'color_segmentation_mask_encoding': 'rgb8',
            'color_palette': [0x556B2F, 0x800000, 0x008080, 0x000080, 0x9ACD32, 0xFF0000, 0xFF8C00,
                              0xFFD700, 0x00FF00, 0xBA55D3, 0x00FA9A, 0x00FFFF, 0x0000FF, 0xF08080,
                              0xFF00FF, 0x1E90FF, 0xDDA0DD, 0xFF1493, 0x87CEFA, 0xFFDEAD],
        }])

    container = ComposableNodeContainer(
        name='unet_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[encoder_node, triton_node, unet_decoder_node],
        output='screen'
    )

    return (launch.LaunchDescription([container]))
