import os
import nibabel as nib
from scipy.ndimage import affine_transform
import SimpleITK as sitk
import logging
import json
import pandas as pd
import subprocess
import ants
import datetime
import time

import shutil
import json
from utils import *

def extract_field_from_sidecar(nifti_path, field_name):
    # Replace .nii.gz extension with .json to get the sidecar file path
    sidecar_path = nifti_path.replace('.nii.gz', '.json')
    # print(f"Side Car: {sidecar_path}")
    # Check if the sidecar file exists
    if not os.path.isfile(sidecar_path):
        # print(f"No JSON sidecar file found at {sidecar_path}")
        return None

    # Open and load the sidecar file
    with open(sidecar_path, 'r') as f:
        sidecar_data = json.load(f)

    # Check if the field exists in the sidecar data
    if field_name not in sidecar_data:
        # print(f"Field '{field_name}' not found in the JSON sidecar file {sidecar_path}")
        return None

    # Return the value of the field
    return sidecar_data[field_name]

    

def parallelize_dataframe(df, func, n_cores=1, log_queue=None):
    import pandas as pd
    import multiprocessing as mp
#     from dipy.align import resample

    df_split = np.array_split(df, n_cores)
    with mp.Pool(n_cores) as pool:
#         df = pd.concat(pool.map(func, df_split))
        df = pd.concat(pool.starmap(func, [(split, log_queue) for split in df_split]))

    return df



def make_isotropic(image, interpolator = sitk.sitkLinear, min_spacing = None):
    '''
    Resample an image to isotropic pixels (using smallest spacing from original) and save to file. Many file formats 
    (jpg, png,...) expect the pixels to be isotropic. By default the function uses a linear interpolator. For
    label images one should use the sitkNearestNeighbor interpolator so as not to introduce non-existant labels.
    '''
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    if min_spacing is None:
        min_spacing = min(original_spacing)
    new_spacing = [min_spacing]*image.GetDimension()
    new_size = [int(round(osz*ospc/min_spacing)) for osz,ospc in zip(original_size, original_spacing)]
    return sitk.Resample(image, new_size, sitk.Transform(), interpolator,
                         image.GetOrigin(), new_spacing, image.GetDirection(), 0,
                         image.GetPixelID())

def nifti_isotropic(input_image_path,output_image_path,label=False,min_spacing = None):
    image = sitk.ReadImage(input_image_path)
    if not label:
        isotropic_image = make_isotropic(image,min_spacing = min_spacing)
    else:
        isotropic_image = make_isotropic(image, interpolator=sitk.sitkNearestNeighbor,min_spacing = min_spacing)
#   for label images use this:
#   
    sitk.WriteImage(isotropic_image, output_image_path)
#   
import shutil

import nibabel as nib
import numpy as np

def mask_volumes(mask_vol, data_vol, output_vol):
    # Load the mask and data volumes
    mask_img = nib.load(mask_vol)
    data_img = nib.load(data_vol)

    # Ensure the mask is binary (0 or 1)
    mask_data = np.where(mask_img.get_fdata() > 0, 1, 0)

    # Apply the mask to the data volume
    masked_data = mask_data * data_img.get_fdata()

    # Create a new NIfTI image with the masked data
    masked_img = nib.Nifti1Image(masked_data, data_img.affine, data_img.header)

    # Save the masked image to the output path
    nib.save(masked_img, output_vol)

# Example usage:
# mask_volumes('path_to_mask.nii.gz', 'path_to_data.nii.gz', 'path_to_output.nii.gz')


def gather_nifti_data(root_directory):
    data = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.nii.gz'):
                file_path = os.path.join(dirpath, filename)
                # print(file_path)
                nifti_img = nib.load(file_path)
                dimensions = nifti_img.shape
                data.append({'patientID':os.path.basename(dirpath),
                             'file_name': filename,
                             'dimensions': dimensions,
                             'file_path': file_path,
                             'XRayExposure':extract_field_from_sidecar(file_path,"XRayExposure"),
                             'AcquisitionTime':extract_field_from_sidecar(file_path,"AcquisitionTime"),
                             'SeriesDescription':extract_field_from_sidecar(file_path,"SeriesDescription"),
                             'AcquisitionNumber':extract_field_from_sidecar(file_path,"AcquisitionNumber")})
    return pd.DataFrame(data)



def proess_row_performRegistrations(row,log_queue):

    try:
#     return 1
        file_name = row['file_name']
        file_path = row["file_path"] # "./vessel_seg2/model_weights/aneurysmDetection/ExtractedData/Tr0028.nii.gz"
        template = "./vessel_seg2/model_weights/atlases/rectangle_neck_scene_RegistrationMask/template_with_skull.nii.gz"
        roi_template = "./vessel_seg2/model_weights/atlases/rectangle_neck_scene_RegistrationMask/skull_neck.nii.gz"
        analysisDir = row["analysisDir"] # r"./vessel_seg2/model_weights/aneurysmDetection/ProcessedData/"
        inferenceDir = row["inferenceDir"]  # "./vessel_seg2/model_weights/aneurysmDetection/InferenceData/"
        inferenceOutDir = row["inferenceOutDir"] # "./vessel_seg2/model_weights/aneurysmDetection/Predictions/"
        mode= row["Mode"]
        overwrite = 1
        log_to_queue(log_queue, f"Registering row {file_name}")

        # step 1 file paths
        outputPrefix = f"{analysisDir}{file_name}_affine_"
        outputWarp = f"{analysisDir}{file_name}_affine_warpedImage.nii.gz"
        warpingTransform_syn = f"{analysisDir}{file_name}_affine_1Warp.nii.gz"
        warpingTransform_aff = f"{analysisDir}{file_name}_affine_0GenericAffine.mat"
        outputWarp_roi = f"{analysisDir}{file_name}_roi_SyN.nii.gz"

        # Step 3 file paths
        cta_roi = f"{analysisDir}{file_name}_cta_inference.nii.gz"
        cta_roi_iso = f"{inferenceDir}{file_name}_0000.nii.gz"

        # 1.	Register head ROI with h+n image
        fixedVolumePath = file_path
        movingVolumePath = template
        antsRegCall = f"antsRegistration \
        --dimensionality 3 \
        --float 0 \
        --output [ {outputPrefix}, {outputWarp} ] \
        --interpolation Linear \
        --winsorize-image-intensities [0.005,0.995] \
        --use-histogram-matching 0 \
        --initial-moving-transform [{fixedVolumePath},{movingVolumePath},1] \
        --transform Rigid[0.1] \
        --metric MI[{fixedVolumePath},{movingVolumePath},1,32,Regular,0.25] \
        --convergence [1000x500x250x100,1e-6,10] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox \
        --transform Affine[0.1] \
        --metric MI[{fixedVolumePath},{movingVolumePath},1,32,Regular,0.25] \
        --convergence [1000x500x250x100,1e-6,10] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox" 
        ants_SynSuffix = f" \
        --transform SyN[0.1,3,0] \
        --metric MI[{fixedVolumePath},{movingVolumePath},1,4] \
        --convergence [100x70x50x20,1e-6,10] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox"

        if mode == "Affine":
            antsRegCommand = antsRegCall
        else: 
            antsRegCommand = antsRegCall + ants_SynSuffix

        log_to_queue(log_queue, f"Performing Template registration for {file_name}")
    #     process = subprocess.Popen(antsRegCommand, stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
    #     output, error = process.communicate()
    #     regExit = process.returncode
        
        if not os.path.isfile(cta_roi_iso) and overwrite:
            regExit = execute_and_log(antsRegCommand,log_queue)
        else:
            log_to_queue(log_queue, f"Found inference image for {file_name}. Moving to the next image.")
            return 0
        log_to_queue(log_queue, f"Template {mode} regisration for {file_name} has return code: {regExit}")

        # 2.	Transform head ROI to h+n image
        if mode == "Affine":
            antsTransformCommand = f"antsApplyTransforms \
            -d 3 -i {roi_template} \
            -r {file_path} \
            -o {outputWarp_roi} \
            -n NearestNeighbor \
            -t {warpingTransform_aff}"
        else:        
            antsTransformCommand = f"antsApplyTransforms \
            -d 3 -i {roi_template} \
            -r {file_path} \
            -o {outputWarp_roi} \
            -n NearestNeighbor \
            -t {warpingTransform_syn} \
            -t {warpingTransform_aff}"

    #     process = subprocess.Popen(antsTransformCommand, stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
    #     output, error = process.communicate()
    #     transExit = process.returncode
        transExit = execute_and_log(antsTransformCommand, log_queue)
        # 3.	Resample to isotropic space
        log_to_queue(log_queue, f"Resampling to isotropic space for {file_name}.")

        if not regExit and not transExit:

            mask_volumes(outputWarp_roi,file_path,cta_roi)
            nifti_isotropic(cta_roi,cta_roi_iso,min_spacing = 0.468)
            return 0
        else: 
            return 1
    except Exception as e:
        return e
    

# def worker(df):
#     return df.apply(proess_row_performRegistrations, axis=1)


# def proess_row_performRegistrations_logg(row, log_queue):
#     # Instead of logging directly, send the log record to the queue
#     log_record = logging.LogRecord(name='root', level=logging.INFO, pathname=__file__, 
#                                    lineno=0, msg=f"Registering row {row['file_name']}", args=(), exc_info=None)
#     log_queue.put(log_record)
#     # ... (rest of the function remains unchanged)
#     return 1


def worker_registration(df, log_queue):
    return df.apply(process_row_perform_registrations_antspy, axis=1, args=(log_queue,))



def proess_row_performResampling(row,log_queue):

    try:
#     return 1
        file_name = row['file_name']
        file_path = row["file_path"] # "./vessel_seg2/model_weights/aneurysmDetection/ExtractedData/Tr0028.nii.gz"
        inferenceOutDir = row["inferenceOutDir"] # "./vessel_seg2/model_weights/aneurysmDetection/Predictions/"
        
        cta_vessel_mask_raw = f"{inferenceOutDir}{row['file_name_internal']}"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The input CTA image {file_path} not found resampling to patient space.")
        if not os.path.exists(cta_vessel_mask_raw):
            raise FileNotFoundError(f"The predicted vessel mask {cta_vessel_mask_raw} does not exist.")
        print(f"{row['cta_predicted']}")
        resamplingCommand = f"antsApplyTransforms -d 3 \
        -i {cta_vessel_mask_raw} \
        -r {file_path} \
        -o {row['cta_predicted']} \
        -n NearestNeighbor \
        -t identity"

        resamplingExit = execute_and_log(resamplingCommand,log_queue)
        
        return resamplingExit
    except FileNotFoundError as e:
        log_to_queue(log_queue, str(e), level=logging.ERROR)
        raise 
    except Exception as e:
        log_to_queue(log_queue, f"Unexpected error during resampling to patient space: {e}", level = logging.ERROR)
        raise
    
def worker_resampling(df, log_queue):
    return df.apply(proess_row_performResampling, axis=1, args=(log_queue,))



def register_images(fixed_image_path, moving_image_path, output_prefix, registration_settings_path, log_queue):
    try:
        if os.path.isfile(output_prefix+"0GenericAffine.mat"):
            log_to_queue(log_queue, f"Affine transformation matrix for {fixed_image_path} found. Skipping to the transformation step.")
            return output_prefix+"0GenericAffine.mat", 55
        
        log_to_queue(log_queue, f"Starting affine registration for {fixed_image_path}")
        with open(registration_settings_path, 'r') as json_file:
            fixed_args = json.load(json_file)
        
        fixed_image = ants.image_read(fixed_image_path)
        moving_image = ants.image_read(moving_image_path)
        
        start_time = time.time()
        registration_dict = ants.registration(fixed=fixed_image,moving=moving_image,
                                         type_of_transform= fixed_args["type_of_transform"], 
                                         outprefix=output_prefix, 
                                         flow_sigma = fixed_args["flow_sigma"],
                                         total_sigma = fixed_args["total_sigma"],
                                         aff_metric = fixed_args["aff_metric"],
                                         aff_iterations = tuple(fixed_args["aff_iterations"]),
                                         aff_shrink_factors = tuple(fixed_args["aff_shrink_factors"]),
                                         aff_smoothing_sigmas = tuple(fixed_args["aff_smoothing_sigmas"]),
                                         verbose = fixed_args["verbose"])
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time

        log_to_queue(log_queue, f"The registration took {elapsed_time} seconds to complete.")
        
        return output_prefix+"0GenericAffine.mat", elapsed_time
    except Exception as e:
        log_to_queue(log_queue, f"Error during registration: {e}", level = logging.ERROR)
        raise
        
def apply_transform(fixed_image_path, moving_image_path, tranform_mat_file, output_image_path, log_queue):
    try:
        fixed_image = ants.image_read(fixed_image_path)
        moving_image = ants.image_read(moving_image_path)
        registeredImage = ants.apply_transforms(fixed_image, moving_image, tranform_mat_file)
        registeredImage.to_file(output_image_path)
    except Exception as e:
        log_to_queue(log_queue, f"Error while applying registration: {e}", level = logging.ERROR)
        raise
    

def mask_and_resample_images(output_warp_roi, file_path, cta_roi, cta_roi_iso, log_queue,mask = True):
    try:
        if mask and output_warp_roi is None:
            log_to_queue(log_queue, f"Masking {file_path} with ROI.")
            mask_volumes(output_warp_roi, file_path, cta_roi)
        else:
            cta_roi = file_path
        log_to_queue(log_queue, f"Resampling to isotropic space for {file_path}.")
        nifti_isotropic(cta_roi, cta_roi_iso, min_spacing=0.468)
    except Exception as e:
        log_to_queue(log_queue, f"Error during image masking and resampling: {e}", level = logging.ERROR)
        raise

def process_row_perform_registrations_antspy(row, log_queue):
    try:
        cta_name = row['file_name']
        cta_path = row["file_path"]
        analysis_dir = row["analysisDir"]
        registration_settings_path = row['registration_settings_path']
#         cta_name_internal = row['']
        template = "./vessel_seg2/model_weights/atlases/rectangle_neck_scene_RegistrationMask/template_with_skull.nii.gz"
        roi_template = "./vessel_seg2/model_weights/atlases/rectangle_neck_scene_RegistrationMask/skull_neck.nii.gz"
        
        

        output_prefix = f"{analysis_dir}{cta_name}_"
        output_warp_roi = f"{analysis_dir}{cta_name}_roi.nii.gz"
        cta_roi = f"{analysis_dir}{cta_name}_cta_inference.nii.gz"
        cta_roi_iso = row['cta_roi_iso']
        
        
        if row['mode']=="Prediction":
            log_to_queue(log_queue, f"Resampling CTA {cta_name} into {cta_roi_iso}.")
            mask_and_resample_images(None, cta_path, cta_roi, cta_roi_iso, log_queue,mask = False)
            # Pass a place holder elapsed time and none for the registration dictionary if masking in not required. 
            return 60, None 
        else:
            log_to_queue(log_queue, f"Registering {cta_name}")
            affine_matrix, elapsed_time = register_images(cta_path, template, output_prefix, registration_settings_path, log_queue)
        
            log_to_queue(log_queue, f"Appylying transform on ROI image.")
            apply_transform(cta_path, roi_template, affine_matrix, output_warp_roi,log_queue)        

            log_to_queue(log_queue, f"Masking CTA image and resampling.")
            mask_and_resample_images(output_warp_roi, cta_path, cta_roi, cta_roi_iso, log_queue)
        
        return elapsed_time,affine_matrix
    except Exception as e:
        log_to_queue(log_queue, f"Error processing row {cta_name}: {e}", level = logging.ERROR)
        return None, None
    


import numpy as np
import nibabel as nib

def mutual_information(hgram):
    """Compute mutual information for joint histogram."""
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can compute the mutual information (MI)
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def compute_mutual_information(fixed_image_path, moving_image_path, bins=20):
    """Compute mutual information between two 3D NIfTI images."""
    # Load the images
    fixed_img = nib.load(fixed_image_path).get_fdata()
    moving_img = nib.load(moving_image_path).get_fdata()
    
    # Compute the joint histogram
    hist_2d, _, _ = np.histogram2d(fixed_img.ravel(), moving_img.ravel(), bins=bins)
    
    # Compute the mutual information from the joint histogram
    return mutual_information(hist_2d)

def update_json_file(file_path, field_name, new_value, log_queue):
    try:

        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Update the specific field with the new value
        data[field_name] = new_value

        # Write the updated content back to the JSON file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        log_to_queue(log_queue, f"Error updating registration settings file: {file_path}", level = logging.ERROR)
        
# Example usage:
# mi_score = compute_mutual_information('path_to_fixed_image.nii.gz', 'path_to_moving_image.nii.gz')
#



def process_directory(input_path, output_path):
    
    df = gather_nifti_data(input_path)
#     df["file_name"] = df["file_name"].apply(lambda x: x.replace('.nii.gz', ''))
    
    return df

def process_csv(csv_path):    
    
    # Read the CSV file without column names
    df = pd.read_csv(csv_path, header=None)

    # Rename the first column to 'file_path'
    df.rename(columns={0: 'file_path'}, inplace=True)
    
    return df

def update_input_dataframe_fields(df, output_path,analysisDir, inferenceDir, inferenceOutDir,registration_settings_path_local,mode,log_queue):
    
    try: 
    
        df["file_name"] = df["file_path"].apply(lambda x: os.path.basename(x))
        df["file_name"] = df["file_name"].apply(lambda x: x.replace('.nii.gz', ''))
        df["analysisDir"]  =analysisDir
        df["inferenceDir"]  =inferenceDir
        df["inferenceOutDir"]  =inferenceOutDir
        df["registration_settings_path"] = registration_settings_path_local
        df["mode"]  = mode
        df['file_name_internal_inference'] = [ "CA_" + "{0:05}_0000.nii.gz".format(x) for x in df.index]
        df['file_name_internal'] = [ "CA_" + "{0:05}.nii.gz".format(x) for x in df.index]
        df['cta_roi_iso'] = df.apply(lambda row: f"{row['inferenceDir']}{row['file_name_internal_inference']}", axis = 1)
        df['cta_predicted'] = df.apply(lambda row: f"{row['inferenceOutDir']}{row['file_name']}.nii.gz", axis = 1)
    except Exception as e:
        log_to_queue(log_queue, f"Error updating input dataframe fields.", level = logging.ERROR)
    
    return df