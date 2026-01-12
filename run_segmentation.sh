#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cta2

export folder=$1
export PATH=./vessel_seg2/ants-2.6.3/bin:$PATHss
python ./vessel_seg2/extractVessels.py -d ${folder} ${folder}_segm  -m 'Prediction' -t 16 -s 0.5 -g 0