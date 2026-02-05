#!/bin/bash

# python src/postprocess/remove_by_roi_jisoo.py "./results/cta_rsna_ane"
# python src/froc_overlap_2.py "./results/cta_rsna_ane" base 
# python src/froc_overlap_2.py "./results/cta_rsna_ane" 1
# python src/froc_overlap_2.py "./results/cta_rsna_ane" 3
# python src/froc_overlap_2.py "./results/cta_rsna_ane" 5

python src/postprocess/remove_by_roi_jisoo.py "./results/hospital140"

python src/froc_overlap_2.py "./results/hospital140" base 
python src/froc_overlap_2.py "./results/hospital140" 1
python src/froc_overlap_2.py "./results/hospital140" 3
python src/froc_overlap_2.py "./results/hospital140" 5

python src/postprocess/remove_by_roi_jisoo.py "./results/cmha"
python src/postprocess/remove_by_roi_jisoo.py "./results/external"
python src/postprocess/remove_by_roi_jisoo.py "./results/internal_test"

python src/froc_overlap_2.py "./results/cmha" base 
python src/froc_overlap_2.py "./results/cmha" 1
python src/froc_overlap_2.py "./results/cmha" 3
python src/froc_overlap_2.py "./results/cmha" 5

python src/froc_overlap_2.py "./results/external" base 
python src/froc_overlap_2.py "./results/external" 1
python src/froc_overlap_2.py "./results/external" 3
python src/froc_overlap_2.py "./results/external" 5

python src/froc_overlap_2.py "./results/internal_test" base 
python src/froc_overlap_2.py "./results/internal_test" 1
python src/froc_overlap_2.py "./results/internal_test" 3
python src/froc_overlap_2.py "./results/internal_test" 5