#!/usr/bin/env python3
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
"""
Export the Isaac ROS DOPE network to one ExecuTorch .pte, and write test data.

The network comes from Isaac ROS's own dope_converter.py, so this .pte and the
ONNX file that script writes share one definition. TensorRT takes every operator
it can. With --cuda-ops, the listed operators stay off TensorRT and run on
ExecuTorch's CUDA backend in the same program. Inputs and outputs live in GPU
memory, owned by the caller, so the ROS node passes message storage straight in.

    # with the packages from the README's step 2
    python export_dope.py --converter dope_converter.py --weights Ketchup.pth \
        --output dope.pte --bag quickstart.bag/quickstart.bag_0.db3
"""

import argparse
import importlib.util
from pathlib import Path
import sqlite3
import struct

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig
import numpy as np
import torch
import torch_tensorrt

# isaac_ros_dope_core.launch.py image encoder defaults.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_network(converter, weights):
    spec = importlib.util.spec_from_file_location('dope_converter', converter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.DopeNetwork()
    state = torch.load(weights, map_location='cpu')
    # The released weights were saved from torch.nn.DataParallel.
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    return net.eval().cuda()


def _cdr_string(buf, off):
    off = (off + 3) & ~3
    n = struct.unpack_from('<I', buf, off)[0]
    return bytes(buf[off + 4:off + 3 + n]).decode(), off + 4 + n


def bag_image(bag, topic):
    """Read the first image on a topic of a rosbag2 SQLite file as RGB, without ROS."""
    con = sqlite3.connect(f'file:{bag}?mode=ro', uri=True)
    row = con.execute(
        'select m.data from messages m join topics t on m.topic_id = t.id '
        "where t.name = ? and t.type = 'sensor_msgs/msg/Image' order by m.timestamp limit 1",
        (topic,)).fetchone()
    if row is None:
        raise ValueError(f'no sensor_msgs/Image on {topic} in {bag}')
    raw = row[0]
    if raw[1] != 1:
        raise ValueError('only little-endian CDR is supported')
    buf = memoryview(raw)[4:]
    _, off = _cdr_string(buf, 8)
    off = (off + 3) & ~3
    height, width = struct.unpack_from('<II', buf, off)
    encoding, off = _cdr_string(buf, off + 8)
    off = (off + 1 + 3) & ~3
    step = struct.unpack_from('<I', buf, off)[0]
    off += 4
    if encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f'unsupported encoding {encoding}')
    data = np.frombuffer(buf[off + 4:off + 4 + height * step], dtype=np.uint8)
    image = data.reshape(height, step)[:, :width * 3].reshape(height, width, 3)
    return np.ascontiguousarray(image[:, :, ::-1] if encoding == 'bgr8' else image)


def preprocess(image, rows, cols):
    x = torch.from_numpy(image).permute(2, 0, 1)[None].float().div(255.0)
    x = torch.nn.functional.interpolate(x, size=(rows, cols), mode='bilinear', align_corners=False)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    parser.add_argument('--converter', required=True, help='Isaac ROS dope_converter.py')
    parser.add_argument('--weights', required=True, help='DOPE .pth file')
    parser.add_argument('--output', required=True, help='.pte file to write')
    parser.add_argument('--rows', type=int, default=720)
    parser.add_argument('--cols', type=int, default=1280)
    parser.add_argument('--cuda-ops', default='',
                        help='comma separated operators to keep on the CUDA backend, '
                        'for example torch.ops.aten.cat.default')
    parser.add_argument('--bag', help='rosbag2 .db3 file; writes <output>.input0.bin, '
                        '<output>.expected0.bin from its first image, and inputs and '
                        'expected outputs 1 and 2 from the same image mirrored and '
                        'upside down, for the test client')
    parser.add_argument('--image-topic', default='/image_rect',
                        help='image topic in the bag (the DOPE quickstart bag uses /image_rect)')
    args = parser.parse_args()

    net = load_network(args.converter, args.weights)
    example = torch.randn(1, 3, args.rows, args.cols, device='cuda')
    exported = torch.export.export(net, (example,))
    options = {'torch_executed_ops': {op.strip() for op in args.cuda_ops.split(',')}} \
        if args.cuda_ops else {}
    compiled = torch_tensorrt.dynamo.compile(
        exported, inputs=[example], min_block_size=1, **options)
    config = ExecutorchBackendConfig(
        propagate_device_config=PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True),
        enable_non_cpu_memory_planning=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False),
    )
    data_file = Path(args.output).parent / 'aoti_cuda_blob.ptd'
    # A data file left by an earlier export would look like this one's.
    data_file.unlink(missing_ok=True)
    torch_tensorrt.save(
        compiled, args.output, output_format='executorch', retrace=False,
        partitioners=[CudaPartitioner([CudaBackend.generate_method_name_compile_spec('forward')])],
        backend_config=config)
    print(f'wrote {args.output}')
    if data_file.exists():
        print(f'wrote {data_file}; pass it to the node as data_file_paths')

    if args.bag:
        image = bag_image(args.bag, args.image_topic)
        # Full FP32 reference, so the check does not depend on TF32 rounding.
        torch.backends.cudnn.allow_tf32 = False
        out = Path(args.output)
        # Three different frames let the test client catch a reply that holds an
        # earlier frame's values.
        frames = [image, np.ascontiguousarray(image[:, ::-1]), np.ascontiguousarray(image[::-1])]
        for k, frame in enumerate(frames):
            x = preprocess(frame, args.rows, args.cols)
            with torch.no_grad():
                expected = net(x.cuda()).cpu()
            x.numpy().tofile(out.with_suffix(f'.input{k}.bin'))
            expected.numpy().tofile(out.with_suffix(f'.expected{k}.bin'))
            print(f'wrote {out.with_suffix(f".input{k}.bin")} {list(x.shape)} and '
                  f'{out.with_suffix(f".expected{k}.bin")} {list(expected.shape)}')


if __name__ == '__main__':
    main()
