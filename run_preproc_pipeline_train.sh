conda activate cta 

export path_base="/data/aneurysm/internal_test/internal_test_pg"
export path_og="${path_base}/og"
export path_label_og="${path_base}/og_label"
export path_resampled=${path_og}_0.4 
export path_label_resampled=${path_label_og}_0.4


python src/preprocess/resample_scans.py ${path_og} ${path_label_og}
python src/preprocess/crop_scans.py ${path_resampled} ${path_base}/crop_0.4 
python src/preprocess/crop_scans.py ${path_label_resampled} ${path_base}/crop_0.4_label


python src/preprocess/compute_distance_maps.py /data/aneurysm/external/crop_0.4_vessel_v2 /data/aneurysm/external_crop_0.4_vessel_v2_edt_test 0.5
