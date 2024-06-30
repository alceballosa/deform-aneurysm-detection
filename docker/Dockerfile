FROM nvcr.io/nvidia/pytorch:23.08-py3


# set workdir to /workspace 
WORKDIR /workspace

RUN git clone https://github.com/alceballosa/deform-aneurysm-detection && \
    pip install -r deform-aneurysm-detection/requirements_torch.txt && \
    pip install -r deform-aneurysm-detection/requirements_base.txt && \
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


