# Preparing Deep Learning Models for Isaac ROS

## Obtaining a Pre-trained Model from NGC
The NVIDIA GPU Cloud hosts a [catalog](https://catalog.ngc.nvidia.com/models) of Deep Learning pre-trained models that are available for your development.

1. Use the **Search Bar** to find a pre-trained model that you are interested in working with.

2. Click on the model's card to view an expanded description, and then click on the **File Browser** tab along the navigation bar.

3. Using the **File Browser**, find a deployable `.etlt` file for the model you are interested in.

    > **Note:** The `.etlt` file extension indicates that this model has pre-trained but **encrypted** weights, which means one needs to use the `tao-converter` utility to decrypt the model, as described [below](#using-tao-converter-to-decrypt-the-encrypted-tlt-model-etlt-format).

4. Under the **Actions** heading, click on the **...** icon for the file you selected in the previous step, and then click **Copy `wget` command**.
5. **Paste** the copied command into a terminal to download the model in the current working directory.

## Using `tao-converter` to decrypt the Encrypted TLT Model (`.etlt`) Format
As discussed above, models distributed with the `.etlt` file extension are encrypted and must be decrypted before use via NVIDIA's [`tao-converter`](https://developer.nvidia.com/tao-toolkit-get-started).

`tao-converter` is already included in the Docker images available as part of the standard [Isaac ROS Development Environment](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/docs/dev-env-setup.md).

The per-platform installation paths are described below:

| Platform        | Installation Path                                             | Symlink Path                        |
| --------------- | ------------------------------------------------------------- | ----------------------------------- |
| x86_64          | `/opt/nvidia/tao/tao-converter-x86-tensorrt8.0/tao-converter` | **`/opt/nvidia/tao/tao-converter`** |
| Jetson(aarch64) | `/opt/nvidia/tao/jp5`                                         | **`/opt/nvidia/tao/tao-converter`** |


### Converting `.etlt` to a TensorRT Engine Plan
Here are some examples for generating the TensorRT engine file using `tao-converter`. In this example, we will use the [`PeopleSemSegnet Shuffleseg` model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplesemsegnet/files?version=deployable_shuffleseg_unet_v1.0):

#### Generate an engine file for the `fp16` data type:
   ```bash
   mkdir -p /workspaces/isaac_ros-dev/models && \
      /opt/nvidia/tao/tao-converter -k tlt_encode -d 3,544,960 -p input_2:0,1x3x544x960,1x3x544x960,1x3x544x960 -t fp16 -e /workspaces/isaac_ros-dev/models/peoplesemsegnet_shuffleseg.engine -o argmax_1 peoplesemsegnet_shuffleseg_etlt.etlt
   ```
   > **Note:** The specific values used in the command above are retrieved from the **PeopleSemSegnet** page under the **Overview** tab. The model input node name and output node name can be found in `peoplesemsegnet_shuffleseg_cache.txt` from `File Browser`. The output file is specified using the `-e` option. The tool needs write permission to the output directory.
   >
   > A detailed explanation of the input parameters is available [here](https://docs.nvidia.com/tao/tao-toolkit/text/tensorrt.html#running-the-tao-converter).

#### Generate an engine file for the data type `int8`:
   
   Create the models directory:
   ```bash
   mkdir -p /workspaces/isaac_ros-dev/models
   ```

   Download the calibration cache file:  
   > **Note:** Check the model's page on NGC for the latest `wget` command.

   ```bash
   wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_shuffleseg_unet_v1.0/files/peoplesemsegnet_shuffleseg_cache.txt
   ```

   ```bash
   /opt/nvidia/tao/tao-converter -k tlt_encode -d 3,544,960 -p input_2:0,1x3x544x960,1x3x544x960,1x3x544x960 -t int8 -c peoplesemsegnet_shuffleseg_cache.txt -e /workspaces/isaac_ros-dev/models/peoplesemsegnet_shuffleseg.engine -o argmax_1 peoplesemsegnet_shuffleseg_etlt.etlt
   ```

   > **Note**: The calibration cache file (specified using the `-c` option) is required to generate the `int8` engine file. This file is provided in the **File Browser** tab of the model's page on NGC.
