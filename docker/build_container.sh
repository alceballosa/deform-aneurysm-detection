#!/bin/bash

# Create a temporary build directory

PATH_TEMP=~/workspace/data_linux/temp-build-context

mkdir $PATH_TEMP

# Build the Docker image
sudo docker build --no-cache -t alceballosa/cta-det -f ./docker/Dockerfile $PATH_TEMP

# Remove the temporary build directory
rm -rf $PATH_TEMP
