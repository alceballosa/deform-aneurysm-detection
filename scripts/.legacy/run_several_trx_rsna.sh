  #!/bin/bash


# export id_gpu=0,1
# export family="deform"
# export model="decoder_only_no_rec_input_edt"
# ./src/run_inference_rsna.sh $family $model "0049999" 0.8 $id_gpu
export id_gpu=0,1
export family="aug"
export model="aug_cnn_4l_short_input_edt_fp_strict_fix"
./src/run_inference_rsna.sh $family $model "final" 0.8 $id_gpu

# export family="xaug"
# export model="opaug_cnn_4l_short_input_edt"
# ./src/run_inference_rsna.sh $family $model "final" 0.8 $id_gpu


# export family="xaug"
# export model="aug_cnn_4l_short_input_edt_fp_strict_fix"
# ./src/run_inference_rsna.sh $family $model "final" 0.8 $id_gpu


# export family="fa"
# export model="aug_cnnfa_4l_short_input_edt_fp_strict_fix"
# ./src/run_inference_rsna.sh $family $model "final" 0.8 $id_gpu