# Deformable at





## Environment setup

This guide assumes that CUDA 12.1 is installed and properly added to PATH variables.

```bash
conda create --name=cta python=3.9
pip install -r requirements_torch.txt
pip install -r requirements_base.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

```

## Inference

1. Get vessel segmentation maps by using the segmentation Docker image and then running the relevant notebook.
2. Create the following file structure:
   1. inputs
      1. scans
      2. vessel_edt_v2
      3. models
         1. model_name

