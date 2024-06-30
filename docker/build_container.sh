#!/bin/bash

# Create a temporary build directory

PATH_TEMP=~/temp-build-context

mkdir $PATH_TEMP

# Copy the Dockerfile
cp -r . $PATH_TEMP

# Build the Docker image
sudo docker build -t cta-det -f $PATH_TEMP/cta-det2/docker/Dockerfile $PATH_TEMP/temp-build-context/

# Remove the temporary build directory
rm -rf temp-build-context

sudo docker save -o cta-det.tar vessel_seg:latest

# Command to test the container:
# sudo docker run --gpus all -it --rm -v ~/Data/aneurysmDetection/output_test_from_container:/Data/aneursysmDetection/output_path/ -v ~/Data/aneurysmDetection/singleRun:/Data/aneursysmDetection/input_cta/ --shm-size=8g --ulimit memlock=-1 vessel_seg:latest
