"""
refine faslse positive cases after aneurysm detection model
remove the bounding box overlay with vein segmentation exclude cavernous sinus(cvs)
process:
    regist cvs mask on target image
    find the bounding box of cvs mask
    expand the bounding box to cover bigger area
    overlay vein seg and bouxing box to get the cvs mask in target image 
"""

import glob
import os
import sys
import time

import nibabel as nib
import tqdm
from AntsRegisteration import ANTsRegistration
from NibImage import NibImage


def get_cvs_mask(
    source_img_path,
    source_mask_path,
    regist_img_path,
    vessel_seg_path,
    output_folder_tmp,
    output_path_mask,
    output_path_bbox,
):
    """
    :param source_img_path: template ct image
    :param source_mask_path: template cvs mask
    :param regist_img_path: target image
    :param vessel_seg_path: vessel segmentation path (from vessel seg model) of regist_img_path
    :return cvs_bbox (nibimage format)
    """
    # read vessel seg and aneurysm bbox image
    while True:
        try:
            vessel_img = nib.load(vessel_seg_path)
            # aneurysm_img = nib.load(aneurysm_bbox_path)

            # crop and clip data for registration
            NibCTA = NibImage(regist_img_path)
            cropped_img = NibCTA.crop_cta_volume(
                crop_height=170,
                save=False,
                save_path=os.path.join(output_folder_tmp, "cropped_img.nii.gz"),
            )
            clipped_img = NibCTA.clip_image(
                cropped_img.get_fdata(),
                save=True,
                save_path=os.path.join(output_folder_tmp, "cropped_img.nii.gz"),
            )

            # registration
            ANTs = ANTsRegistration(
                fix_img_path=os.path.join(output_folder_tmp, "cropped_img.nii.gz"),
                move_img_path=source_img_path,
            )
            transformed_mask_img = ANTs.mask_registration(
                move_mask_path=source_mask_path,
                save=False,
                output_path=os.path.join(output_folder_tmp, "regist_cvs_mask.nii.gz"),
            )
            NibCTA.reverse_crop_cta(
                cropped_data=transformed_mask_img.numpy(),
                save=True,
                save_path=os.path.join(output_folder_tmp, "regist_cvs_mask.nii.gz"),
            )

            # cvs_mask = nib.load('./reversed_cropped_mask.nii.gz')
            # cropped_cvs_mask = nib.load('./cvs_mask.nii.gz')
            # vessel_mask = nib.load(vessel_seg_path)
            # data_img = nib.load(regist_img_path)
            # print(cropped_cvs_mask.shape, cvs_mask.shape, vessel_mask.shape, data_img.shape)
            # transformed_mask_img = ants.image_read('./cvs_mask.nii.gz')

            # get bbox for transformed cvs mask and expand in 3 direction
            transformed_mask_data = transformed_mask_img.numpy()
            bbox_img = NibCTA.find_cvs_bbox(transformed_mask_data)
            expand_bbox = NibCTA.expand_bbox(bbox_img.get_fdata())
            expand_bbox = NibCTA.reverse_crop_cta(
                cropped_data=expand_bbox.get_fdata(),
                save=True,
                save_path=os.path.join(output_folder_tmp, "expand_bbox.nii.gz"),
            )

            # get cvs mask
            cvs_mask_img = NibCTA.overlay_vein_mask(
                vessel_img=vessel_img,
                bbox_img=expand_bbox,
                save=True,
                save_path=output_path_mask,
            )

            cvs_bbox_img = NibCTA.find_cvs_bbox(
                cvs_mask_img.get_fdata(),
                save=True,
                save_path=output_path_bbox,
            )
            break
        except:
            print("Failed to compute, re-trying")
    return cvs_bbox_img

    # remove aneurysm bbox which overlap with vein region
    # NibCTA.remove_vein_bbox(vessel_img=vessel_img, cvs_mask_img=cvs_mask_img, aneurysm_bbox_img=aneurysm_img,
    #                        save=True, save_path=os.path.join(output_folder,'aneurysm_bbox.nii.gz'))

    # end_time = time.time()


if __name__ == "__main__":

    # cavernous sinus segmentation path (manually seg by Jisoo)
    source_img_path = "./src/cvs_mask/SyN_iteration9_antsBTPtemplate1.nii.gz"
    source_mask_path = "./src/cvs_mask/CVS-vein-trim.nrrd"

    # output from aneurysm detection model
    scans_folder = sys.argv[1]
    vessel_folder = sys.argv[2]
    output_folder_temp = sys.argv[3]
    output_folder_mask = sys.argv[4]
    output_folder_bbox = sys.argv[5]
    # output folder for cvs
    all_files = glob.glob(output_folder_temp)

    os.makedirs(output_folder_temp, exist_ok=True)
    os.makedirs(output_folder_mask, exist_ok=True)
    os.makedirs(output_folder_bbox, exist_ok=True)

    image_list = os.listdir(scans_folder)
    list_aneurysms = []
    for i in tqdm.tqdm(image_list):
        regist_img_path = os.path.join(scans_folder, i)
        vessel_seg_path = os.path.join(vessel_folder, i)
        output_path_mask = os.path.join(output_folder_mask, i)
        output_path_bbox = os.path.join(output_folder_bbox, i)
        # check if file exists 
        if os.path.isfile(output_path_mask) and os.path.isfile(output_path_bbox):
            print(f"{i} was already done.")
            continue
        os.makedirs(output_folder_temp, exist_ok=True)
        cvs_img = get_cvs_mask(
            source_img_path,
            source_mask_path,
            regist_img_path,
            vessel_seg_path,
            output_folder_temp,
            output_path_mask,
            output_path_bbox,
        )
        print(f"{i} done")
