#!/bin/bash

# Create a temporary build directory

PATH_TEMP=~/workspace/data_linux/temp-build-context

mkdir $PATH_TEMP

# Build the Docker image
sudo docker build --no-cache -t alceballosa/cta-det -f ./docker/Dockerfile $PATH_TEMP

# Remove the temporary build directory
rm -rf temp-build-context

#sudo docker save -o cta-det.tar vessel_seg:latest

# Command to test the container:
# sudo docker run --gpus all -it --rm -v ~/Data/aneurysmDetection/output_test_from_container:/Data/aneursysmDetection/output_path/ -v ~/Data/aneurysmDetection/singleRun:/Data/aneursysmDetection/input_cta/ --shm-size=8g --ulimit memlock=-1 vessel_seg:latest
