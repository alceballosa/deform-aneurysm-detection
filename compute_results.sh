#!/bin/bash

python src/postprocess/remove_by_roi_dilated.py "./results/cmha"
python src/postprocess/remove_by_roi_dilated.py "./results/external"
python src/postprocess/remove_by_roi_dilated.py "./results/hospital140"
python src/postprocess/remove_by_roi_dilated.py "./results/internal_test"

python src/froc_overlap.py "./results/hospital140"
python src/froc_overlap.py "./results/internal_test"
python src/froc_overlap.py "./results/cmha"
python src/froc_overlap.py "./results/external"

