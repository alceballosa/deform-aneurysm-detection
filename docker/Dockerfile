FROM nvcr.io/nvidia/pytorch:23.08-py3


# set workdir to /workspace 
WORKDIR /workspace

# Install dependencies for ANTs
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    gcc \
    git \
    zlib1g-dev

# Clone and build ANTs
RUN git clone https://github.com/ANTsX/ANTs.git /opt/ants && \
    mkdir /opt/ants/build && \
    cd /opt/ants/build && \
    cmake .. && \
    make

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install antspyx

RUN mkdir -p /Data/toshiba/nnUNet_files/nnUNet_results_production/Dataset1050_toshiba_iCafe/nnUNetTrainer__nnUNetPlans__3d_fullres/
RUN mkdir -p /Data/toshiba/nnUNet_files/nnUNet_preprocessed_production/Dataset1050_toshiba_iCafe/
RUN mkdir -p /Data/toshiba/nnUNet_files/nnUNet_raw/Dataset1050_toshiba_iCafe/
RUN mkdir -p /Data/toshiba/nnUNet_files/nnUNet_results/Dataset1150_toshiba_iCafe/nnUNetTrainer__nnUNetPlans__3d_fullres/
RUN mkdir -p /Data/toshiba/nnUNet_files/nnUNet_preprocessed/Dataset1150_toshiba_iCafe/
RUN mkdir -p /Data/toshiba/nnUNet_files/nnUNet_raw/Dataset1150_toshiba_iCafe/

RUN mkdir -p /Work/scripts/
RUN mkdir -p /Data/aneurysmDetection/input_cta
RUN mkdir -p /Data/aneurysmDetection/output_path
RUN mkdir -p /Data/aneurysmDetection/csv_location
RUN mkdir -p /Data/atlases

# Copy the nnU-Net contents into the container at /Data/toshiba/nnUNet_files_production/nnUNet_results/
COPY Dataset1050_toshiba_iCafe/nnUNetTrainer__nnUNetPlans__3d_fullres /Data/toshiba/nnUNet_files/nnUNet_results/Dataset1050_toshiba_iCafe/nnUNetTrainer__nnUNetPlans__3d_fullres
COPY Dataset1150_toshiba_vesselSeg2/nnUNetTrainer__nnUNetPlans__3d_fullres /Data/toshiba/nnUNet_files/nnUNet_results/Dataset1150_toshiba_iCafe/nnUNetTrainer__nnUNetPlans__3d_fullres
COPY Dataset1150_toshiba_vesselSeg2/nnUNetTrainer_NexToU__nnUNetPlans__3d_fullres_nextou /Data/toshiba/nnUNet_files/nnUNet_results/Dataset1150_toshiba_iCafe/nnUNetTrainer_NexToU__nnUNetPlans__3d_fullres_nextou
# copy atlas
COPY atlases /Data/atlases

# Copy the vessel segmentation pipeline into the container at /Work/scripts/

COPY helper.py /Work/scripts/helper.py
COPY utils.py /Work/scripts/utils.py
COPY extractVessels.py /Work/scripts/extractVessels.py

# Add ANTs binaries to PATH
ENV PATH="/opt/ants/bin:${PATH}"
ENV PATH="/opt/ants/Scripts:/opt/ants/build/ANTS-build/Examples:${PATH}"


RUN mkdir /workspace/NexToU
# Clone NexToU repository
RUN git clone https://github.com/PengchengShi1220/NexToU.git /workspace/NexToU

# Copy NexToU components to nnUNet (adjust paths as necessary)
RUN cp /workspace/NexToU/loss/* /nnUNet/nnunetv2/training/loss/ && \
    cp /workspace/NexToU/network_architecture/* /nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/ && \
    cp /workspace/NexToU/nnUNetTrainer/* /nnUNet/nnunetv2/training/nnUNetTrainer/
    
RUN pip install edt

