# ExecuTorch inference node for Isaac ROS

This package runs a model with [ExecuTorch](https://pytorch.org/executorch/),
PyTorch's solution for deploying models on devices. The node uses the same
message type, topics and tensor name parameters as the Isaac ROS TensorRT node,
so it can replace that node in a graph. Unlike that node, it matches the named
tensors to the program's inputs and outputs in order (it has no binding name
parameters), and each input must have exactly the shape the program was
exported with. It uses ExecuTorch's TensorRT and CUDA backends.

## Why

Today a model reaches the TensorRT node as an ONNX file, with
`custom_plugin_lib` for operators TensorRT cannot convert, or it goes to the
Triton node. With ExecuTorch:

- The graph from `torch.export` goes straight to TensorRT, with no ONNX file.
- Standard PyTorch (ATen) operators that Torch-TensorRT cannot convert stay in
  the same program and run on the GPU through ExecuTorch's CUDA backend, when
  PyTorch's compiler can generate GPU code for them. This was tested with one
  such operator, `erfinv` (step 8). Custom CUDA kernels are not covered.
- The C++ node links only the ExecuTorch and TensorRT libraries. It does not
  load PyTorch.

What it costs:

- The node brings its own libraries: ExecuTorch, and TensorRT 11.4 with its CUDA
  runtime, from pip. Isaac ROS 5.1 uses TensorRT 10.16 from apt.
- A `.pte` holds TensorRT engines that are already built for one GPU type and
  one TensorRT version. So you export one `.pte` per GPU type and TensorRT
  version, on a machine with PyTorch and that GPU.
- A program with CUDA backend parts needs a full CUDA toolkit for the export,
  which compiles those parts. When they load, they need a libstdc++ from GCC
  13.2 or newer (see Limits).

DOPE, the model in the steps below, has no operator that TensorRT cannot
convert, so TensorRT takes the whole network and there is no speed gain. The
mixed DOPE program in step 5 reaches the CUDA backend only by forcing
concatenation off TensorRT. The `erfinv` model in step 8 shows the real case.

## The node

`nvidia::isaac_ros::dnn_inference::ExecuTorchNode` subscribes to `tensor_pub`
and publishes `tensor_sub`, both `isaac_ros_tensor_msgs/TensorList`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model_file_path` | `model.pte` | The ExecuTorch program |
| `data_file_paths` | `[]` | `.ptd` files; pass every one the export wrote |
| `input_tensor_names` | `['input_tensor']` | Message tensor names, in the program's input order |
| `output_tensor_names` | `['output']` | Names of the reply tensors, in the program's output order |
| `input_qos`, `output_qos` | `DEFAULT` | QoS profiles, as in the TensorRT node |
| `input_queue_size`, `output_queue_size` | `1` | Queue depths, as in the TensorRT node |

When an input is in a CUDA buffer (the CUDA buffer backend of
`rosidl::Buffer`), the node reads it in place, and the program writes its
outputs straight into the reply's CUDA buffer, so no data goes through the CPU.
This needs a program whose inputs and outputs are on the GPU, as both export
scripts make it: `ExecutorchBackendConfig` with
`PropagateDeviceConfig(skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True)`,
`enable_non_cpu_memory_planning=True` and
`MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False)`. For a
program that keeps them in host memory, the node logs a warning and copies
through the CPU. An input in host memory is copied to the GPU once. All GPU
work runs on one CUDA stream that the node owns. Like the other Isaac ROS
nodes, it turns on intra-process communication, so in one container the image
encoder's CUDA buffer reaches it as it is. A message without data, or with
another name, shape or data type than the program takes, is skipped with no
reply, and logged at most once per second.

To put it into an Isaac ROS graph, load it into the same container as the other
nodes:

```python
ComposableNode(
    package='isaac_ros_executorch',
    plugin='nvidia::isaac_ros::dnn_inference::ExecuTorchNode',
    name='dope_inference',
    parameters=[{
        'model_file_path': model,
        'input_tensor_names': ['input_tensor'],
        'output_tensor_names': ['output'],
    }],
)
```

## Requirements

- An x86_64 Linux machine with an NVIDIA GPU and a driver that supports CUDA 13
  (version 580 or newer). The steps below were run on an H100 with driver 580.
- Python 3.11 or newer with the `venv` module (`python3-venv` on Ubuntu; 3.12
  was tested), `git`, `curl`, and `ar` and `xz` to unpack the VPI packages
  (`binutils` and `xz-utils` on Ubuntu).
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
  to install ROS 2 Lyrical and a CUDA 13.2 toolkit from
  [RoboStack](https://robostack.github.io/) conda packages. The node needs ROS 2
  Lyrical for `rosidl::Buffer`, like Isaac ROS 5.1. Only Lyrical was tested.
- These ROS packages, which step 1 clones:
  - `isaac_ros_common` and `isaac_ros_tensor_msgs` from
    [isaac_ros_common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common)
    release `v5.1-0`. Outside an Isaac ROS container, `isaac_ros_common` needs
    NVIDIA VPI 4, which step 6 downloads.
  - `cuda_buffer_backend` and `tensor_msgs/ExperimentalTensor` from
    [rosidl_buffer_backends](https://github.com/ros2/rosidl_buffer_backends),
    the same CUDA buffer backend Isaac ROS 5.1 uses.
  - `executorch_conversions`, `executorch_conversions_cuda`,
    `executorch_conversions_cpu` and `executorch_cuda_vendor`. These are not
    merged yet. They are in the open pull request
    [#36](https://github.com/ros2/rosidl_buffer_backends/pull/36). Step 1 checks
    out the commit of that pull request that was tested, which also holds
    `cuda_buffer_backend` and `tensor_msgs`.

## Try it

All steps run in one directory, `~/isaac_et`, in one terminal. Only the test
client in steps 7 and 8 needs a second terminal. Start from a shell that has no
other ROS 2 installation sourced, so that only the conda one is used.

**1. Get the sources.**

```bash
mkdir -p ~/isaac_et/ws/src && cd ~/isaac_et
git clone https://github.com/shoumikhin/rosidl_buffer_backends.git ws/src/rosidl_buffer_backends
git -C ws/src/rosidl_buffer_backends fetch origin 8e429d30b01982fc2021bc6285e2de53063de06f
git -C ws/src/rosidl_buffer_backends checkout 8e429d30b01982fc2021bc6285e2de53063de06f
git clone -b v5.1-0 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git ws/src/isaac_ros_common
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git ws/src/isaac_ros_dnn_inference
```

**2. Torch-TensorRT with ExecuTorch**, in a fresh virtual environment, for the
export. The ROS build in step 6 takes only the ExecuTorch and TensorRT C++
libraries from it, and the node loads them from it when it runs, so keep the
environment in place. One `pip install` keeps torchvision from changing the
torch version. The versions are the tested ones; they keep ExecuTorch at 1.6,
which the pinned rosidl_buffer_backends commit is built for.

```bash
python3 -m venv venv
venv/bin/pip install "torch_tensorrt[executorch]==2.15.*" "torchvision==0.29.*" "onnx==1.23.*"
```

**3. The model and a test image.** `dope_converter.py` is Isaac ROS's own DOPE
network definition, from the `v5.1-0` release of isaac_ros_pose_estimation.
`Ketchup.pth` holds the DOPE weights from the
[DOPE weights folder](https://drive.google.com/drive/folders/1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg).
The quickstart bag comes from the Isaac ROS DOPE assets on NGC, version 5.0.0.

```bash
curl -LO https://raw.githubusercontent.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/v5.1-0/isaac_ros_dope/scripts/dope_converter.py
curl -L -o Ketchup.pth "https://drive.usercontent.google.com/download?id=1FBuPqzsSJHEqmxT7XW2vknuUDi2sF3XY&confirm=t"
curl -L -o quickstart.tar.gz https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/isaac_ros_dope_assets/versions/5.0.0/files/quickstart.tar.gz
sha256sum -c <<'EOF' && tar -xzf quickstart.tar.gz
78c8a940cbdcc6c6b5bd718c77974d7d408c0617e41d38701e65a12276cf04af  dope_converter.py
d63a4ea651d92dd91185bdc9543232fd71fb2950cf0742d1f6178e4f5d6fe80f  Ketchup.pth
0b7564100ca043edbdd68d11013984ed930a0f0ffe0968e69f16151be5eb5a37  quickstart.tar.gz
EOF
```

**4. ROS 2 Lyrical and the CUDA 13.2 toolkit.** Set `CONDA_OVERRIDE_CUDA` to
the CUDA version that `nvidia-smi` shows for your driver. The 13.0 below is
what the tested driver 580 reports.

```bash
CONDA_OVERRIDE_CUDA=13.0 micromamba create -y -p ~/isaac_et/ros -c conda-forge -c robostack-lyrical \
  python=3.12 ros-lyrical-ros-base ros-lyrical-sensor-msgs ros-lyrical-std-msgs \
  ros-lyrical-rmw-fastrtps-cpp ros-lyrical-rosidl-buffer ros-lyrical-rosidl-buffer-backend \
  ros-lyrical-rosidl-buffer-backend-registry ros-lyrical-rosidl-buffer-py \
  ros-lyrical-pluginlib ros-lyrical-rclcpp-components ros-lyrical-launch-ros \
  ros-lyrical-ament-cmake-auto ros-lyrical-rosidl-default-generators \
  ros-lyrical-ament-cmake-gtest ros-lyrical-ament-cmake-ros ros-lyrical-ament-lint-auto \
  colcon-common-extensions cmake ninja compilers pkg-config pytest git \
  cuda-version=13.2 cuda-nvcc cuda-cudart-dev cuda-driver-dev cuda-nvtx-dev
eval "$(micromamba shell hook -s bash)"
micromamba activate ~/isaac_et/ros
```

**5. Export the model**, on a machine with the GPU type you deploy to.
ExecuTorch compiles the CUDA backend parts while it exports, so `CUDA_HOME`
points at the CUDA 13.2 toolkit from step 4. Its headers and libraries are in
the `targets` directory of the environment. The conda environment puts its own
Python packages on `PYTHONPATH`; `-E` keeps them out of the virtual environment.

```bash
export CUDA_HOME=$CONDA_PREFIX/targets/x86_64-linux
mkdir -p dope dope_cuda
venv/bin/python -E ws/src/isaac_ros_dnn_inference/isaac_ros_executorch/scripts/export_dope.py \
  --converter dope_converter.py --weights Ketchup.pth --output dope/dope.pte \
  --bag isaac_ros_dope/quickstart.bag/quickstart.bag_0.db3
venv/bin/python -E ws/src/isaac_ros_dnn_inference/isaac_ros_executorch/scripts/export_dope.py \
  --converter dope_converter.py --weights Ketchup.pth --output dope_cuda/dope.pte \
  --cuda-ops torch.ops.aten.cat.default
```

The first command writes `dope/dope.pte`. TensorRT takes the whole DOPE network.
`--bag` also writes three test frames from the quickstart image (the image, its
mirror image, and the image upside down), with the expected outputs, for the
check in step 7. The network definition comes from `dope_converter.py`, so the
`.pte` and the Isaac ROS ONNX file share one definition.

The second command makes a mixed program: `--cuda-ops` keeps the listed
operators off TensorRT, and they run on the CUDA backend. A program with a
CUDA backend part also writes an `aoti_cuda_blob.ptd` data file next to it, so
give each program its own directory. Pass that file to the node
(`data_file_paths`): a CUDA part whose weights are missing still loads, without
an error.

**6. Build the ROS packages**, with the CUDA 13.2 toolkit from step 4 and NVIDIA
VPI 4 from NVIDIA's apt repository, unpacked locally.

```bash
B=https://repo.download.nvidia.com/jetson/x86_64/noble
mkdir -p vpi
curl -L -o vpi/libnvvpi4.deb $B/pool/main/libn/libnvvpi4/libnvvpi4_4.1.3_amd64.deb
curl -L -o vpi/vpi4-dev.deb $B/pool/main/v/vpi4-dev/vpi4-dev_4.1.3_amd64.deb
sha256sum -c <<'EOF' &&
bbea8f20455a1efef605a8367470888e367b8ae1c2bc49218cc7507640326d1d  vpi/libnvvpi4.deb
7a732bf5d0eeee272a137cb752d249df82fa4590823014d645b4ec9fed1780f8  vpi/vpi4-dev.deb
EOF
for d in libnvvpi4 vpi4-dev; do
  mkdir -p vpi/$d && (cd vpi/$d && ar x ../$d.deb && tar -xf data.tar.* -C ..)
done
SP=$(venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
cd ws
colcon build --merge-install --packages-up-to isaac_ros_executorch --cmake-args \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_CUDA_ARCHITECTURES=native \
  -Dvpi_DIR=$HOME/isaac_et/vpi/opt/nvidia/vpi4/lib/x86_64-linux-gnu/cmake/vpi \
  -Dexecutorch_DIR=$SP/executorch/share/cmake \
  -Dexecutorch_backend_tensorrt_DIR=$SP/torch_tensorrt_executorch_runtime/lib/cmake/executorch_backend_tensorrt
cd ..
```

**7. Run.** Start the node with the TensorRT-only program:

```bash
source ws/install/setup.bash
ros2 launch isaac_ros_executorch isaac_ros_executorch.launch.py model_file_path:=$PWD/dope/dope.pte
```

In a second terminal, send the recorded frames and check the replies:

```bash
cd ~/isaac_et
eval "$(micromamba shell hook -s bash)"
micromamba activate ~/isaac_et/ros
source ws/install/setup.bash
ros2 run isaac_ros_executorch tensor_list_client --ros-args \
  -p "input_files:=[$PWD/dope/dope.input0.bin,$PWD/dope/dope.input1.bin,$PWD/dope/dope.input2.bin]" \
  -p "expected_files:=[$PWD/dope/dope.expected0.bin,$PWD/dope/dope.expected1.bin,$PWD/dope/dope.expected2.bin]" \
  -p "input_shape:=[1,3,720,1280]" -p "output_shape:=[1,25,90,160]" \
  -p runs:=200
```

To run the mixed program, stop the node and start it with the data file. The
client command stays the same, because both programs compute the same output:

```bash
ros2 launch isaac_ros_executorch isaac_ros_executorch.launch.py model_file_path:=$PWD/dope_cuda/dope.pte data_file_paths:=$PWD/dope_cuda/aoti_cuda_blob.ptd
```

**8. A model with an operator TensorRT cannot take.** DOPE runs fully in
TensorRT. This small model (convolution, `erfinv`, convolution) shows the CUDA
backend at work: Torch-TensorRT has no converter for `erfinv`, so the export
puts the convolutions in TensorRT and `erfinv` in the CUDA backend. The script
prints the parts and writes three test frames.

```bash
export CUDA_HOME=$CONDA_PREFIX/targets/x86_64-linux
mkdir -p erfinv
venv/bin/python -E ws/src/isaac_ros_dnn_inference/isaac_ros_executorch/scripts/export_erfinv.py --output erfinv/model.pte
```

Start the node with it, in the first terminal:

```bash
ros2 launch isaac_ros_executorch isaac_ros_executorch.launch.py model_file_path:=$PWD/erfinv/model.pte data_file_paths:=$PWD/erfinv/aoti_cuda_blob.ptd
```

Check it from the second terminal:

```bash
ros2 run isaac_ros_executorch tensor_list_client --ros-args \
  -p "input_files:=[$PWD/erfinv/model.input0.bin,$PWD/erfinv/model.input1.bin,$PWD/erfinv/model.input2.bin]" \
  -p "expected_files:=[$PWD/erfinv/model.expected0.bin,$PWD/erfinv/model.expected1.bin,$PWD/erfinv/model.expected2.bin]" \
  -p "input_shape:=[1,3,64,64]" -p "output_shape:=[1,9,64,64]" \
  -p runs:=200 -p tolerance:=0.001
```

The client waits for the reply with each request's frame id. A reply fails when
it does not arrive within `timeout_s` (5 s by default), lacks the output tensor,
has another shape or data type, is not in a CUDA buffer although the request
was, holds a value that is not finite, or differs from the expected values by
more than the tolerance. The client cycles through the frames, so a reply that
holds an earlier frame's values fails too. It prints one line with the counts, the largest error and `PASS`
or `FAIL`, and exits nonzero on `FAIL`, also when a parameter or a file is
wrong. The default tolerance, 0.01, leaves room for TensorRT's TF32 math against
PyTorch in full FP32 precision.

## Tests

`colcon test` runs the ROS 2 linters and a launch test. The launch test loads
`test/affine.pte`, a small CPU program that computes `2 * x + 1`, into the node.
It checks the reply values, and that the node skips a message without data and
a message without its input tensor, and keeps serving. `test/export_affine.py` writes
that file. With the environment from "Try it" active:

```bash
cd ~/isaac_et/ws
colcon build --merge-install --packages-select isaac_ros_executorch --cmake-args -DBUILD_TESTING=ON
colcon test --merge-install --packages-select isaac_ros_executorch
colcon test-result --verbose
```

## Results

From the steps above on an H100, with every message in a CUDA buffer:

| Program | Replies correct | Largest error |
| --- | --- | --- |
| DOPE, TensorRT only | 200 of 200 | 0.0024 |
| DOPE, TensorRT + CUDA (with its .ptd) | 200 of 200 | 0.0026 |
| erfinv (TensorRT, CUDA, TensorRT) | 200 of 200 | 1.2e-7 |

The error is against PyTorch in full FP32 precision. The node process maps no
PyTorch library.

## Limits

Before this can merge:

- `executorch_conversions`, `executorch_conversions_cpu`,
  `executorch_conversions_cuda` and `executorch_cuda_vendor` exist only in the
  open pull request ros2/rosidl_buffer_backends#36, and have no rosdep keys. The
  TensorRT backend (`executorch_backend_tensorrt`) comes from a pip wheel and
  has no ROS package, and the build records the path of the virtual environment
  in the node's library search path. So `rosdep install` and a build of the
  whole repository fail until #36 is released and a ROS package for the TensorRT
  backend exists (planned in
  [pytorch/executorch#24819](https://github.com/pytorch/executorch/issues/24819)).
- This node uses TensorRT 11.4 from pip and Isaac's TensorRT node uses TensorRT
  10.16 from apt. Most functions these TensorRT libraries export have the same
  names, with no symbol versions, so run the nodes in separate processes.
  Running both in one process was not tested.

Other limits:

- The steps in "Try it" are for x86_64 with conda. On Jetson the node was built
  against the Isaac ROS 5.1 apt packages and run there, not inside the Isaac ROS
  Docker image.
- Intra-process delivery skips the check of which buffer types a subscriber
  accepts, so a node in the same container that accepts only CPU buffers may get
  CUDA buffers. The other Isaac ROS nodes run the same way. This case was not
  tested.
- The node handles one program on one GPU (`cuda:0`), with fixed input and output shapes
  and row major tensors. It does not load a program whose inputs or outputs are
  planned by the program or are in another dim order (for example channels
  last). A program whose output shares memory with an input or another output
  loads, but the node skips every message.
- The program must not change its inputs in place. The node hands it the
  received message's storage, which other subscribers may share, so such a
  program would change their data too. The node cannot check this at load.
- The CUDA backend parts of a `.pte` exported this way are compiled with the
  conda compilers (GCC 15) and need `GLIBCXX_3.4.32` (libstdc++ from GCC 13.2 or
  newer). The node gets libstdc++ from the conda environment. Ubuntu 24.04,
  which the Isaac ROS 5.1 Docker image uses, ships GCC 14's libstdc++, but the
  image was not tested.
- On x86_64 with glibc 2.41 or newer (for example Ubuntu 26.04), a program with
  CUDA backend parts exported with ExecuTorch 1.6 fails to load, because its
  compiled code asks for an executable stack (pytorch/pytorch#203463). With
  glibc 2.42 or newer, `GLIBC_TUNABLES=glibc.rtld.execstack=2` allows it.
- The CUDA backend takes only operators that PyTorch's compiler generates GPU
  code for. An operator the compiler hands back to a PyTorch library kernel is
  rejected at export (`Method forward missing fallback kernels`).
- ExecuTorch's CUDA backend writes a small shared library per CUDA part to
  `/tmp` when the program loads, and removes it when the node stops normally. A
  node killed with SIGKILL leaves `/tmp/executorch_cuda_<pid>_<n>.so` behind.
  Some state of ExecuTorch's CUDA backend also stays in the process after the
  node is unloaded or fails to load, so restart the container rather than
  loading the node again into the same one.

## License

Apache-2.0. See [LICENSE](../LICENSE).
