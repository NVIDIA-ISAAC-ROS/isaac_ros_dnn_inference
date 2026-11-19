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
Write affine.pte, the CPU program the launch test runs: y = 2 * x + 1.

    # with the packages from the README's step 2
    python export_affine.py --output affine.pte
"""

import argparse

from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.passes import MemoryPlanningPass
import torch


class Affine(torch.nn.Module):

    def forward(self, x):
        return 2 * x + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    parser.add_argument('--output', required=True, help='.pte file to write')
    args = parser.parse_args()

    exported = torch.export.export(Affine(), (torch.zeros(2, 3),))
    program = to_edge(exported).to_executorch(ExecutorchBackendConfig(
        # The node hands message storage to the program, so inputs and outputs are not planned.
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False),
    ))
    with open(args.output, 'wb') as file:
        program.write_to_file(file)


if __name__ == '__main__':
    main()
