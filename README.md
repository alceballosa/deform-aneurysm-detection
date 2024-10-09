# Vessel-aware aneurysm detection using multi-scale deformable 3D attention

Paper URL: https://papers.miccai.org/miccai-2024/831-Paper2366.html

Vessel segmentation docker image can be downloaded from: https://drive.google.com/file/d/1qa91P423Sp5fMqUUoMxBGV0EUEAQirBC/view 

Weights can be downloaded from the following link: https://drive.google.com/file/d/1-5gOZEcdJ14Ght1hSZGSKsAyPLFT-y8o/view?usp=sharing

After doing so, create the following folders under the repository folder:

`models/decoder_only_no_rec_pe_edt/`

And place the weights file inside.

## Environment setup (non-Docker)

This guide assumes that CUDA 12.1 is installed and properly added to PATH variables.

```bash
conda create --name=cta python=3.10
conda activate cta
pip install -r requirements_torch.txt
pip install -r requirements_base.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python==4.8.0.76
```

## Vessel segmentation + EDT generation

First, run the provided Docker image as follows to get the vessel segmentations:
```
sudo docker run --gpus all -it --rm -v [PATH_TO_SEGM_OUTPUTS]/:/Data/aneurysmDetection/output_path/  -v [PATH_TO_SCANS]/:/Data/aneurysmDetection/input_cta/ --shm-size=24g --ulimit memlock=-1 vessel_seg:latest python /Work/scripts/extractVessels.py -d /Data/aneurysmDetection/input_cta/ /Data/aneurysmDetection/output_path -m 'Prediction' -t 16 -s 1 -g 1"
```

Then, run the following notebook making sure to define the relevant paths: `notebooks/scan_processing/get_distance_maps.ipynb`, saving the outputs to [PATH_TO_VESSEL_DISTANCE_MAPS].



## Inference
Finally, run inference as follows:
```bash
MODEL_NAME=decoder_only_no_rec_pe_edt
CHECKPOINT_NAME="final"
SCAN_DIR="[PATH_TO_SCANS]"
VESSEL_DIR="[PATH_TO_VESSEL_DISTANCE_MAPS]"
./run_inference_docker.sh $MODEL_NAME $CHECKPOINT_NAME DATA.DIR.VAL.SCAN_DIR $SCAN_DIR DATA.DIR.VAL.VESSEL_DIR $VESSEL_DIR 
```

