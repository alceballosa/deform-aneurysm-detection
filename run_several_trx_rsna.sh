  #!/bin/bash


export id_gpu=0,1
export family="trx"
export model="cnn_4l_input_edt_amp_ogopt"
./src/run_inference_rsna.sh $family $model "final" 0.8 $id_gpu


export family="xaug"
export model="opaug_cnn_4l_short_input_edt"
./src/run_inference_rsna.sh $family $model "final" 0.8 $id_gpu


export family="xaug"
export model="aug_cnn_4l_short_input_edt_fp_strict_fix"
./src/run_inference_rsna.sh $family $model "final" 0.8 $id_gpu


export family="fa"
export model="aug_cnnfa_4l_short_input_edt_fp_strict_fix"
./src/run_inference_rsna.sh $family $model "final" 0.8 $id_gpu