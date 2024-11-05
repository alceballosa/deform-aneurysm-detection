# Vessel-aware aneurysm detection using multi-scale deformable 3D attention

Work accepted to MICCAI 2024.

Weights and sample files can be downloaded from: WIP

## Environment setup (non-Docker)

Install CUDA toolkit 12.1.0 as follows or use an existing installation:

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

Then install the following dependencies:

```bash
conda create --name=cta python=3.10
conda activate cta
pip install -r requirements_torch.txt
pip install -r requirements_base.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python==4.8.0.76
```

## Inference (non-Docker)

```bash
MODEL_NAME=decoder_only_no_rec_pe_edt_v2
CHECKPOINT_NAME="0065999"
SCAN_DIR="/workspace/inputs/scans"
VESSEL_DIR="/workspace/inputs/vessel_edt_v2"
./run_inference_docker.sh $MODEL_NAME $CHECKPOINT_NAME DATA.DIR.VAL.SCAN_DIR $SCAN_DIR DATA.DIR.VAL.VESSEL_DIR $VESSEL_DIR 
```
DATA:
  PATCH_SIZE: [64, 64, 64]
  OVERLAP: [32, 32, 32]
  N_CHANNELS: 1
  DIR:
    VAL:
      SCAN_DIR: "/workspace/inputs/scans"
      VESSEL_DIR: "/workspace/inputs/vessel_edt_v2"


## Docker image for inference

1. Get vessel segmentation maps by using the segmentation Docker image and then running the relevant notebook.
2. Create the following file structure:
    1. inputs
        1. scans
        2. vessel_edt_v2
        3. models
            1. model_name

Build the docker image as follows:

```bash
chmod +x ./docker/build_container.sh
./docker/build_container.sh
```

And run inference as follows (change variables as required):

```bash
PATH_INPUTS=./inputs/
PATH_OUTPUTS=./outputs/
MODEL_NAME=decoder_only_no_rec_pe_edt_v2
CHECKPOINT_NAME="0065999"
sudo docker run --gpus all -it --rm -v  $PATH_INPUTS:/workspace/inputs -v $PATH_OUTPUTS:/workspace/deform-aneurysm-detection/outputs  --shm-size=32g --ulimit memlock=-1 alceballosa/cta-det:latest  ./deform-aneurysm-detection/run_inference_docker.sh $MODEL_NAME $CHECKPOINT_NAME
```



