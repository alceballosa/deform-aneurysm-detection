# Vessel-aware aneurysm detection using multi-scale deformable 3D attention

Work accepted to MICCAI 2024. 

## Environment setup

This guide assumes that CUDA 12.1 is installed and properly added to PATH variables.

```bash
conda create --name=cta python=3.9
pip install -r requirements_torch.txt
pip install -r requirements_base.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

```

## Inference


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


And run inference as:
```bash
PATH_INPUTS=./inputs/
PATH_OUTPUTS=./outputs/
MODEL_NAME=decoder_only_no_rec_pe_edt_v2
CHECKPOINT_NAME="0065999"
sudo docker run --gpus all -it --rm -v  $PATH_INPUTS:/workspace/inputs -v $PATH_OUTPUTS:/workspace/deform-aneurysm-detection/outputs  --shm-size=32g --ulimit memlock=-1 alceballosa/cta-det:latest  ./deform-aneurysm-detection/run_inference_docker.sh $MODEL_NAME $CHECKPOINT_NAME


`` 
