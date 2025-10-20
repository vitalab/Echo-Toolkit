# Echo-Toolkit

Code repository for an echocardiography medical imaging toolkit.
Functionalities also extend to general ultrasound imaging.

We ask that you cite the `Domain Adaptation of Echocardiography Segmentation Via Reinforcement Learning` paper, which made use of the sector extraction tool, in exchange for the use of this project:

```
@InProceedings{Jud_Domain_MICCAI2024,
        author = { Judge, Arnaud and Judge, Thierry and Duchateau, Nicolas and Sandler, Roman A. and Sokol, Joseph Z. and Bernard, Olivier and Jodoin, Pierre-Marc},
        title = { { Domain Adaptation of Echocardiography Segmentation Via Reinforcement Learning } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
        year = {2024},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15009},
        month = {October},
        page = {235-244},
}
```

Current functionalities:
- Echocardiography metrics for 2-3D segmentation (BG, LV and MYO)
- Ultrasound sector extraction tool

# Echo Metrics
Echo Toolkit provides a set of evaluation metrics to assess the quality and consistency of echocardiographic segmentations. These metrics include **Dice, Hausdorff distance, anatomical validity, temporal consistency**, and **mitral valve localization**.

To obtain all of these metrics, use the `full_test_metrics` function from `echotk.metric.test_metrics`.

### Quick Reference
- **Dice Score** (`dice`)  
  Measures overlap between predicted and ground-truth segmentations for each label (LV, MYO, BG).

- **Hausdorff Distance (HD)** (`hausdorff`)  
  Measures boundary mismatch between predicted and ground-truth segmentations in physical units (mm).

- **Anatomical Validity** (`is_anatomically_valid`)  
  Checks whether each frame’s segmentation is anatomically plausible.

- **Temporal Consistency** (`check_temporal_validity`)  
  Detects abrupt changes or inconsistencies across consecutive frames using LV/MYO area, epicardial center, LV length, and base width.

- **Mitral Valve Distance** (`mitral_valve_distance`)  
  Computes frame-wise errors in mitral valve localization relative to the ground-truth, reporting MAE, MSE, and thresholded mistake rates per cardiac cycle.

- **Temporal Hausdorff Metric** (`get_temporal_hd_metric`, `temporal_hd`)  
  Measures boundary differences across consecutive frames to assess smoothness over time.

- **Cycle Analysis** (`extract_cycle_points`, `estimate_num_cycles`)  
  Detects systolic peaks and diastolic valleys in LV area and estimates the number of cardiac cycles.

### Example Usage

```python
from echotk.metrics.test_metrics import full_test_metrics

# batchwise_3d_segmentation: predicted 3D segmentation (T, H, W) -- in batch format (Time first)
# batchwise_gt: ground-truth 3D segmentation
# voxel_spacing: pixel spacing in mm
# device: torch device (cpu or cuda)

logs = full_test_metrics(batchwise_3d_segmentation, batchwise_gt, voxel_spacing, device, prefix="test", verbose=True)

# Access specific metrics
print("Dice LV:", logs["test/dice/LV"])
print("HD MYO:", logs["test/hd/MYO"])
print("Anatomical validity:", logs["test/anat_valid"])
print("Temporal errors:", logs["test/temporal_errors"])
```

** Some metrics based on code originally from https://github.com/vitalab/vital

# Ultrasound sector extraction
### Usage notes
Inference time may be quite long if using an older GPU and long input sequences. 
Included test examples (which are small) for sector extraction require up to 2-3 seconds per prediction (using roughly 7GB of VRAM), using a NVIDIA RTX3090 GPU.

## Usage through Docker container

Images for this project are available here: https://hub.docker.com/r/arnaudjudge/echo-toolkit.

In order to use a Docker container with the host machine's GPU to run this tool, you must install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The `--gpus all` flag allows for use of the host GPU.

To run through the Docker images, use the following command:
```bash
  sudo docker run -it --ipc host --gpus all -v $(pwd)/:/ETK_MOUNT/ --user $(id -u):$(id -g) arnaudjudge/echo-toolkit:latest etk_extract_sector input=/ETK_MOUNT/<PATH_TO_INPUT_FILE> output=/ETK_MOUNT/<PATH_TO_OUTPUT>
```

The command syntax is as follows:
- `--ipc host` gives the container more shared memory.
- `--gpus all` allocates access to all gpus on host machine. `--gpus "device=0"` can be used to allocate a specific gpu only.
- `-v $(pwd)/:/ETK_MOUNT/` mounts the current directory to the /ETK_MOUNT/ directory in the container, allowing for file syncing between host and container.
- `--user $(id -u):$(id -g)` allows to user in the container to be the same as outside it. Output files will not be locked by sudo user once created.
- `input=/ETK_MOUNT/<PATH_TO_INPUT_FILE>` input file. <u>The input file (or folder) must be within the current mounted directory.</u>*
- `output=/ETK_MOUNT/<PATH_TO_OUTPUT>` indicated the output file location. <u>The output folder location must be within the current mounted directory.</u>*

\* Input and output paths must be located in current directory in order for files to be visible to the container and to be synced. This is true for any pah referenced in command line arguments.

To use the extraction tool without use of a GPU, use this command instead:
```bash
  sudo docker run -it --ipc host -v $(pwd)/:/ETK_MOUNT/ --user $(id -u):$(id -g) arnaudjudge/echo-toolkit:latest etk_extract_sector input=/ETK_MOUNT/<PATH_TO_INPUT_FILE> output=/ETK_MOUNT/<PATH_TO_OUTPUT> accelerator=cpu
```
The `accelerator=cpu` argument changes the pytorch accelerator. Keep in mind that inference times can be very long with use of CPUs only. 

Debugging in the container can be done with the following command, opening a bash command line in the container:
```bash
  sudo docker run -it --ipc host --gpus all -v $(pwd)/:/ETK_MOUNT/ --user $(id -u):$(id -g) arnaudjudge/echo-toolkit:latest bash
```

## Install
To run fully locally install the project and its dependencies:

1. Download the repository:
   ```bash
   # clone project
   git clone --recurse-submodules https://github.com/vitalab/Echo-Toolkit.git
   cd Echo-Toolkit
   ```
2. Create a virtual environment (Conda is strongly recommended):
   ```bash
   # create conda environment
   conda create -n echotk python=3.10
   conda activate echotk
   ```
3. Install [PyTorch](https://pytorch.org/get-started/locally/) according to instructions. Grab the one with GPU for faster training:
   ```bash
   # example for linux or Windows
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
4. Install the project in editable mode and its dependencies, do the same for the ASCENT submodule:
   ```bash
   pip install -e .
   cd ASCENT
   pip install -e .
   cd ..
   ```

## Usage
### Ultrasound sector extraction

Ultrasound sector extraction removes all annotations, text, ECG trace and unnecessary information from the images, 
keeping only the region of interest, the *cone* or *sector*.

|![gif before extraction](./assets/a4c.gif)<br>Before|![gif after extraction](./assets/a4c_masked.gif)<br>After|
|:-:|:-:|


This project contains a command line script to create and apply a mask to remove all such annotations. 
To run it, simply use the following command:

```bash
etk_extract_sector
```

By default, it will process input data from the `./data/examples/` folder at the project's root. 
Example images and sequences are included already [^1].

Many options are available through hydra (https://hydra.cc/docs/intro/) CLI override syntax or 
through modification/addition of config files. 
The main config files with all available default arguments is `./echotk/config/sector_extract.yaml`.

Examples of command line usage:
```bash
etk_extract_sector input=<NEW_PATH_TO_DATA>
etk_extract_sector input=<NEW_PATH_TO_DATA> output=<PATH_TO_OUTPUT> show_result_gifs=True save_metrics=True
```

The `input` argument must be a folder, file or list of files. Files are expected to be in nifti format.
By default, outputs files are send to `./output`.

## Implementation details
### Ultrasound sector extraction

The sector extraction tool is based on RANSAC. RANSAC (Random sample consesus) is an iterative algorithm 
to estimate parameters for a mathematical model, in this case lines and circles.
Based on points extracted from the edge of the ultrasound cone, two line models and one circle model are fit, 
then the resulting mask is reconstructed based on their intersections.

In order to improve performance, a nn-UNet model was trained (using the ASCENT project included as a submodule). 
This model predicts a cone for the input sequence of images. 
This cone is used to obtain an initial version of the sector mask.
Since the model has no enforced prior about the sector's shape, there can be inconsistencies in its shape.
RANSAC is therefore applied to the edge points of the prediction to obtain a reliable output mask.
We found that using the neural network before RANSAC rather than RANSAC alone (on variance metrics of input sequence) 
made for a much more robust method.

[^1] Images originally from Wikimedia Commons, available under Creative Commons License at https://commons.wikimedia.org/w/index.php?curid=25498405.