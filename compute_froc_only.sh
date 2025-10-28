#!/bin/bash

python src/postprocess/remove_by_roi_jisoo.py "./results/hospital140"

python src/froc_overlap.py "./results/hospital140" 0
python src/froc_overlap.py "./results/hospital140" base 
python src/froc_overlap.py "./results/hospital140" 1
python src/froc_overlap.py "./results/hospital140" 2
python src/froc_overlap.py "./results/hospital140" 3
python src/froc_overlap.py "./results/hospital140" 4
python src/froc_overlap.py "./results/hospital140" 5

python src/postprocess/remove_by_roi_jisoo.py "./results/cmha"

python src/froc_overlap.py "./results/cmha" 0
python src/froc_overlap.py "./results/cmha" base 
python src/froc_overlap.py "./results/cmha" 1
python src/froc_overlap.py "./results/cmha" 2
python src/froc_overlap.py "./results/cmha" 3
python src/froc_overlap.py "./results/cmha" 4
python src/froc_overlap.py "./results/cmha" 5


python src/postprocess/remove_by_roi_jisoo.py "./results/external"
python src/postprocess/remove_by_roi_jisoo.py "./results/internal_test"






python src/froc_overlap.py "./results/external" 0
python src/froc_overlap.py "./results/external" base 
python src/froc_overlap.py "./results/external" 1
python src/froc_overlap.py "./results/external" 2
python src/froc_overlap.py "./results/external" 3
python src/froc_overlap.py "./results/external" 4
python src/froc_overlap.py "./results/external" 5

python src/froc_overlap.py "./results/internal_test" 0
python src/froc_overlap.py "./results/internal_test" base 
python src/froc_overlap.py "./results/internal_test" 1
python src/froc_overlap.py "./results/internal_test" 2
python src/froc_overlap.py "./results/internal_test" 3
python src/froc_overlap.py "./results/internal_test" 4
python src/froc_overlap.py "./results/internal_test" 5