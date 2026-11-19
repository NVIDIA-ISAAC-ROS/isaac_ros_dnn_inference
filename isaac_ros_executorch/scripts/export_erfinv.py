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
Export a small model that Torch-TensorRT cannot convert in full, and write test data.

The model is a convolution, erfinv, and a convolution. Torch-TensorRT has no
converter for erfinv, so the convolutions go to TensorRT and erfinv is left
over; ExecuTorch's CUDA backend takes it, in the same program. The script
prints the parts and writes three test frames with their expected outputs.

    # with the packages from the README's step 2
    python export_erfinv.py --output erfinv/model.pte
"""

import argparse
from pathlib import Path

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig
import torch
import torch_tensorrt


class ConvErfinvConv(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.second = torch.nn.Conv2d(8, 9, 1)

    def forward(self, x):
        # tanh keeps the erfinv input inside (-1, 1).
        return self.second(torch.special.erfinv(torch.tanh(self.first(x)) * 0.9))


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    parser.add_argument('--output', required=True, help='path of the .pte to write')
    args = parser.parse_args()

    torch.manual_seed(0)
    net = ConvErfinvConv().eval().cuda()
    example = torch.randn(1, 3, 64, 64, device='cuda')
    exported = torch.export.export(net, (example,))
    compiled = torch_tensorrt.dynamo.compile(exported, inputs=[example], min_block_size=1)
    for name, part in compiled.named_children():
        if hasattr(part, 'graph'):
            ops = sorted({str(n.target) for n in part.graph.nodes if n.op == 'call_function'})
            print(f'{name}: left over by Torch-TensorRT: {ops}')
        else:
            print(f'{name}: TensorRT engine')
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

    # Full FP32 reference, so the check does not depend on TF32 rounding.
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output)
    for k in range(3):
        x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(100 + k))
        with torch.no_grad():
            expected = net(x.cuda()).cpu()
        x.numpy().tofile(out.with_suffix(f'.input{k}.bin'))
        expected.numpy().tofile(out.with_suffix(f'.expected{k}.bin'))
    print(f'wrote 3 test frames {list(x.shape)} -> {list(expected.shape)}')


if __name__ == '__main__':
    main()
