exp_dict = {
    "deform_decoder_only_4_layers": "Non-Rec 4 Layers",
    "deform_decoder_only_unet_lr": "Non-Rec U-Net",
    "deform_decoder_only_input_96": "Non-Rec Input 96",
    "deform_decoder_only_non_rec_BEST_cropinf": "Non-Rec (Base)",
    "deform_decoder_only_128_points": "Non-Rec 128 offset points",
    "deform_decoder_only_rec_shared_step_lr": "Rec 2 Steps",
    "deform_decoder_only_rec_shared_step_lr_3_layers": "Rec 3 Steps",
    "deform_decoder_only_rec_shared_step_lr_4_layers": "Rec 4 Steps",
    "deform_decoder_only_non_rec_ns_paper_lr": "Non-Rec 3D Offset Lr=2e-4",
    "deform_decoder_only_rec_shared_step_lr_med_bsz": "Rec 2 Layers Step Med BSZ (Bad)",
    "deform_decoder_only_non_rec_crop": "Rec 2 Layers Crop",
    "deform_decoder_only_non_rec_good_med_bsz": "Non-Rec Med Bsz (Good)",
    "deform_decoder_only_non_rec_BEST_cropinf_TI": "Non-Rec (Base) [TRAIN]",
    "deform_decoder_only_rec_shared_step_lr_4_layers_TI": "Rec 4 Steps [TRAIN]",
    "deform_decoder_only_4_layers_TI": "Non-Rec 4 Layers [TRAIN]",
    "deform_decoder_only_4_layers_EXT": "Non-Rec 4 Layers [EXT]",
    "deform_decoder_only_rec_shared_step_lr_4_layers_EXT": "Rec 4 Steps [EXT]",
    "deform_decoder_only_non_rec_BEST_cropinf_EXT": "Non-Rec (Base) [EXT]",
    "deform_decoder_only_rec_shared_step_lr_5_layers": "Rec 5 Steps",
    "deform_decoder_only_rec_shared_step_lr_5_layers_TI": "Rec 5 Steps [TRAIN]",
    "deform_decoder_only_rec_shared_step_lr_5_layers_EXT": "Rec 5 Steps [EXT]",
    "deform_decoder_only_non_rec_16_heads_random": "Non-Rec Random Init",
    "deform_decoder_only_non_rec_16_heads_relaxed": "Non-Rec Relaxed Init",
    "deform_decoder_only_non_rec_16_heads_random_EXT": "Non-Rec Random Init [EXT]",
    "deform_decoder_only_non_rec_16_heads_relaxed_EXT": "Non-Rec Relaxed Init [EXT]",
    "adeform_decoder_only_non_rec_crop_vessel_start_gpe": "Non-Rec Vessel Start",
    "adeform_decoder_only_non_rec_crop_vessel_start_gpe_EXT": "Non-Rec Vessel Start [EXT]",
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe": "Non-Rec Vessel PE",
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_EXT": "Non-Rec Vessel PE [EXT]",
    "adeform_decoder_only_rec_4_layers_crop_unet_vessel_pe_gpe": "Rec 4 Steps UNET Vessel PE",
    "adeform_decoder_only_rec_4_layers_crop_unet_vessel_pe_gpe_EXT": "Rec 4 Steps UNET Vessel PE [EXT]",
    "adeform_decoder_only_non_rec_crop_unet_vessel_pe_gpe": "Non-Rec UNET Vessel PE",
    "adeform_decoder_only_non_rec_crop_unet_vessel_pe_gpe_EXT": "Non-Rec UNET Vessel PE [EXT]",
}


exp_names = [
    "deform_decoder_only_4_layers",  # 0
    "deform_decoder_only_unet_lr",  # 1
    "deform_decoder_only_input_96",  # 2
    "deform_decoder_only_non_rec_BEST_cropinf",  # 3
    "deform_decoder_only_128_points",  # 4
    "deform_decoder_only_rec_shared_step_lr",  # 5
    "deform_decoder_only_rec_shared_step_lr_3_layers",  # 6
    "deform_decoder_only_rec_shared_step_lr_4_layers",  # 7
    "deform_decoder_only_non_rec_ns_paper_lr",  # 8
    "deform_decoder_only_rec_shared_step_lr_med_bsz",  # 9
    "deform_decoder_only_non_rec_crop",  # 10
    "deform_decoder_only_non_rec_good_med_bsz",  # 11
    "deform_decoder_only_non_rec_BEST_cropinf_TI",  # 12
    "deform_decoder_only_rec_shared_step_lr_4_layers_TI",  # 13
    "deform_decoder_only_4_layers_TI",  # 14
    "deform_decoder_only_4_layers_EXT",
    "deform_decoder_only_rec_shared_step_lr_4_layers_EXT",
    "deform_decoder_only_non_rec_BEST_cropinf_EXT",
    "deform_decoder_only_rec_shared_step_lr_5_layers",  # 18
    "deform_decoder_only_rec_shared_step_lr_5_layers_TI",  # 19
    "deform_decoder_only_rec_shared_step_lr_5_layers_EXT",  # 19
    "deform_decoder_only_non_rec_16_heads_random",  # 20
    "deform_decoder_only_non_rec_16_heads_relaxed",  # 21
    "deform_decoder_only_non_rec_16_heads_random_EXT",  # 22
    "deform_decoder_only_non_rec_16_heads_relaxed_EXT",  # 23
    "adeform_decoder_only_non_rec_crop_vessel_start_gpe",
    "adeform_decoder_only_non_rec_crop_vessel_start_gpe_EXT",
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe",
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_EXT",

]
viz_names = [
    "Non-Rec 4 Layers",
    "Non-Rec U-Net",
    "Non-Rec Input 96",
    "Non-Rec (Base)",
    "Non-Rec 128 offset points",
    "Rec 2 Steps",
    "Rec 3 Steps",
    "Rec 4 Steps",
    "Non-Rec 3D Offset Lr=2e-4",
    "Rec 2 Layers Step Med BSZ (Bad)",
    "Rec 2 Layers Crop",
    "Non-Rec Med Bsz (Good)",
    "Non-Rec (Base) [TRAIN]",
    "Rec 4 Steps [TRAIN]",
    "Non-Rec 4 Layers [TRAIN]",
    "Non-Rec 4 Layers [EXT]",
    "Rec 4 Steps [EXT]",
    "Non-Rec (Base) [EXT]",
    "Rec 5 Steps",
    "Rec 5 Steps [TRAIN]",
    "Rec 5 Steps [EXT]",
    "Non-Rec Random Init",
    "Non-Rec Relaxed Init",
    "Non-Rec Random Init [EXT]",
    "Non-Rec Relaxed Init [EXT]",
    "Non-Rec Vessel Start GPE",
    "Non-Rec Vessel Start GPE [EXT]",
    "Non-Rec Vessel PE GPE",
    "Non-Rec Vessel PE GPE [EXT]",
]


def get_names():
    return exp_names, viz_names


def get_names_dict():
    return exp_dict
