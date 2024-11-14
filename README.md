# Vessel-aware aneurysm detection using multi-scale deformable 3D attention

![Figure describing the model](diagram.png)

Paper URL: https://papers.miccai.org/miccai-2024/831-Paper2366.html

Vessel segmentation docker image can be downloaded from: https://drive.google.com/file/d/1qa91P423Sp5fMqUUoMxBGV0EUEAQirBC/view 

You then can run ```docker load -i vessel_seg.tar``` to make it available in your environment.

Weights can be downloaded from the following link: https://drive.google.com/file/d/1-5gOZEcdJ14Ght1hSZGSKsAyPLFT-y8o/view?usp=sharing

After doing so, create the following chain of folders inside the repository:

`models/decoder_only_no_rec_pe_edt/`

And place the weights file inside.

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

## Inference instructions

### File organization for inference

You should organize your input files as follows:

```bash
[ROOT_FOLDER]/
    og/
        scan_1.nii.gz
        ...
        scan_n.nii.gz
```

### Preprocessing files for inference

Now, change the path defined in line 9 of ```run_preproc_pipeline_inf.sh``` to your root folder and run it as follows:

```bash
./run_preproc_pipeline_inf.sh
```

### Running inference

After running the above, you can do inference using the following list of commands. Just make sure you replace [ROOT_FOLDER] with the actual path to your data.

Note: we define threshold to be the same as used in our paper but feel free to change it if you want slightly higher sensitivity at the cost of more false positives.
```bash

export path_base="[ROOT_FOLDER]"
export path_base="/data/aneurysm/test"
export path_scans="${path_base}/crop_0.4"
export path_edt="${path_base}/crop_0.4_vessel_edt"
export path_outputs="${path_base}/predictions"
export model_name="decoder_only_no_rec_pe_edt"
export checkpoint_name="final"
export threshold=0.95

./run_inference.sh ${model_name} ${checkpoint_name} ${path_scans} ${path_edt} ${path_outputs} ${threshold}
```

Output files will be placed under the ```predictions``` folder.


## Training instructions

WIP.
