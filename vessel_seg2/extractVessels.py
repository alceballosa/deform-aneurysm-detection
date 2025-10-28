import argparse
import datetime
import json
import logging
import multiprocessing as mp
import os
import shutil
import textwrap
import time

import pandas as pd
from helper import *

from utils import *

# from data_processing import process_csv, process_directory, setup_output_directories
# from pipeline import run_pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
    A deep learning tool to create a binary mask of arteries from a high resolution CTA image. Refer to the manuscript for the specifics. Familiarity with the ANTs registration toolbox will be helpful but not necessary. 
    
    usage with a directory: 
        python extractVessels.py -d ./vessel_seg2/model_weights/aneurysmDetection/ExtractedDataTest ./vessel_seg2/model_weights/aneurysmDetection/Outputs_antsPy/
    usage with a csv containing nii.gz file paths: 
        python extractVessels.py -c ./vessel_seg2/model_weights/aneurysmDetection/cta_input_list.csv ./vessel_seg2/model_weights/aneurysmDetection/Outputs_antsPy/"""
        ),
        epilog=textwrap.dedent(
            """\
    outputs: 
     This command will create many directories in the output location. The relevant files/directories are as follows:
         output_path/Predictions - The nnU-Net model predictions will be saved here. Files containing the vessel mask in patient space will be saved with the same name as the input CTA images.
         output_path/IntermediateFiles - The files created during the registration steps will be saved here. 
         output_path/InferenceData - The resampled CT images that will be used by the nnU-net model will be saved here. 
         output_path/vesselSegmentation_{PIPELINE_MODE}_{dateTime}.log - This is the log file of the pipeline. Each run will create its own log file. Share this file with the original author for debugging. 
         output_path/qc_{PIPELINE_MODE}_{dateTime}.csv - csv file containing the registration quality control metrics for all the registrations in a particular run.
         output_path/pipeline_failures_{PIPELINE_MODE}_{dateTime}.csv - (TODO) csv file containing the file paths of registrations or segmetations that failed along with the reason of the failure. 
         output_path/antspy_registration_settings.json - the json file containing the arguements that run the registration steps. Modify the settings here only if the affine registrations fail in the first or second pass. Refer to https://github.com/ANTsX/ANTs/wiki/Tips-for-improving-registration-results#debugging-registration-failures for help on debugging the registration pipeline."""
        ),
    )

    # Input path argument
    #     parser.add_argument("input_path", type=str, help="Path to the input CTA data.")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "-c",
        "--csv",
        type=str,
        help="Path to the CSV file containing list of CTA NIfTI image paths.",
    )
    group.add_argument(
        "-d",
        "--directory",
        type=str,
        help="Path to the directory containing CTA NIfTI files.",
    )

    # Output path argument
    parser.add_argument("output_path", type=str, help="Path to save the vessel mask.")

    # Number of threads argument
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use for brain ROI extraction.",
    )

    # Registration
    parser.add_argument(
        "-r",
        "--registration_method",
        type=str,
        default="Affine",
        help="Registration method to use in Ants. Currently tested for two options: 'Affine' and 'AffineFast'. Use the latter for testing the pipeline.",
    )

    # Number of GPUs argument
    parser.add_argument(
        "-g",
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for inference.",
    )

    # Registration mode
    parser.add_argument(
        "-m",
        "--pipeline_mode",
        type=str,
        default="Full",
        help="This flag should either be 'ROI', 'Prediction' or 'Full'. 'ROI' mode only affine registers the \
                        tempalte to the CT image and creates a masked image with only the head+neck for future use. Useful \
                        when you want to split the processing over multiple machines or commands.\
                        'Prediction' mode directly prepares the input CTA for prediction without ROI extraction and creates vessel masks using the nnU-Net model.\
                        'Full' mode performs both registration and vessel extraction on the input CT dataset.",
    )

    parser.add_argument(
        "-s",
        "--sliding_window",
        type=float,
        default=0.25,
        help="Sliding window for nnU-Net inference. Use -s 1 for testing the pipeline.",
    )

    args = parser.parse_args()

    return args


def setup_output_directories(output_path):
    # Create output directory structure
    analysisDir = output_path + "/IntermediateFiles/"
    inferenceDir = output_path + "/InferenceData/"
    inferenceOutDir = output_path + "/Predictions/"

    if not os.path.exists(analysisDir):
        os.makedirs(analysisDir)
    if not os.path.exists(inferenceDir):
        os.makedirs(inferenceDir)
    if not os.path.exists(inferenceOutDir):
        os.makedirs(inferenceOutDir)

    return analysisDir, inferenceDir, inferenceOutDir


def runPipeline(
    df,
    output_path,
    num_threads,
    num_gpus,
    log_queue,
    pipeline_failures_csv,
    qc_csv,
    mode,
    sliding_window,
):

    analysisDir, inferenceDir, inferenceOutDir = setup_output_directories(output_path)

    log_to_queue(log_queue, f"There are {len(df)} CTA samples in all.")

    log_to_queue(log_queue, f"Running the pipieline using {num_threads} threads.")
    df["worker_registration"] = parallelize_dataframe(
        df, worker_registration, n_cores=num_threads, log_queue=log_queue
    )
    df[["elapsed_time", "affine_matrix"]] = df["worker_registration"].apply(pd.Series)
    df = df.dropna(subset=["elapsed_time"])

    filtered_df = df[df["elapsed_time"] < 10]
    if len(filtered_df) > 0:
        filtered_df["failure_message"] = "Suspiciously low elapsed time."
        # Write the 'file_path' column of the filtered DataFrame to a CSV file
        filtered_df[["file_path", "failure_message"]].to_csv(
            pipeline_failures_csv, index=False
        )

        log_to_queue(
            log_queue,
            f"File paths with elapsed time less than 10 seconds have been written to {pipeline_failures_csv}",
        )
    else:
        log_to_queue(log_queue, f"No registration took less than 10 seconds.")

    if mode == "ROI":
        log_to_queue(
            log_queue,
            "Pipeline completed. Pass -m 'Full' or -m 'Prediction' for the inference.",
        )
        return

    log_to_queue(
        log_queue, f"Starting the inference on all avaiable files under {inferenceDir}."
    )
    inferenceCommand = f"nnUNetv2_predict --continue_prediction -c 3d_fullres \
    -i '{inferenceDir}' \
    -o '{inferenceOutDir}' \
    -d 1150  -f 0 -step_size {sliding_window} -chk checkpoint_best.pth -device cuda"

    inferenceExit = execute_and_log(inferenceCommand, log_queue)

    log_to_queue(log_queue, f"Inference completed. Exit code: {inferenceExit}")

    log_to_queue(log_queue, "Starting the resampling step.")
    df["ResamplingExit"] = parallelize_dataframe(
        df, worker_resampling, n_cores=num_threads, log_queue=log_queue
    )
    for result in df["ResamplingExit"]:
        if isinstance(result, Exception):
            log_to_queue(
                log_queue,
                f"An error occurred in a worker process: {result}",
                level=logging.ERROR,
            )

    num_ones = df["ResamplingExit"].sum()
    num_zeros = len(df) - num_ones

    log_to_queue(log_queue, f"Number of Successful Resamplings: {num_zeros}")

    if num_ones > 0:
        log_to_queue(
            log_queue,
            f"Warning: There were unsuccessful ROI transformations for {num_ones} samples.",
            level=logging.WARNING,
        )
        log_to_queue(
            log_queue,
            f"Warning: Look at the intermediate files for the following patientIDs: {num_ones}",
            level=logging.WARNING,
        )

    df.to_csv(
        qc_csv,
        sep=",",
        header=True,
        index=False,
        doublequote=False,
        columns=["file_path", "elapsed_time", "ResamplingExit"],
    )

    return


def main():

    args = parse_arguments()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_file = os.path.join(
        args.output_path, f"vessel_segmentation_{args.pipeline_mode}_{current_time}.log"
    )
    print(f"Setting up logging process. Refer to {log_file}")

    manager = mp.Manager()
    log_queue = manager.Queue()

    listener = mp.Process(target=listener_process, args=(log_queue, log_file))
    listener.start()

    try:

        log_to_queue(log_queue, "Setting environemnt variables.")
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = (
            "4"  # found to not impact the performance
        )
        os.environ["nnUNet_raw"] = "./vessel_seg2/model_weights/toshiba/nnUNet_files/nnUNet_raw/"
        os.environ["nnUNet_preprocessed"] = (
            "./vessel_seg2/model_weights/toshiba/nnUNet_files/nnUNet_preprocesssed/"
        )
        os.environ["nnUNet_results"] = "./vessel_seg2/model_weights/toshiba/nnUNet_files/nnUNet_results/"

        # read in registration settings for the registration pipeline
        registration_settings_path = "./vessel_seg2/model_weights/atlases/rectangle_neck_scene_RegistrationMask/antspy_registration_settings.json"

        registration_settings_path_local = os.path.join(
            args.output_path, "antspy_registration_settings.json"
        )

        if not os.path.isfile(registration_settings_path_local):
            log_to_queue(
                log_queue,
                "Did not find ants registration json. Using default settings.",
            )
            shutil.copy2(registration_settings_path, registration_settings_path_local)
            if args.registration_method is not None:
                log_to_queue(
                    log_queue,
                    "Updating registration settings file with the supplied registration method.",
                )
                update_json_file(
                    registration_settings_path_local,
                    "type_of_transform",
                    args.registration_method,
                    log_queue,
                )
        else:
            log_to_queue(
                log_queue,
                f"Found ants registration json. Ensure the arguments are correct for obtaining correct registration. Delete {registration_settings_path_local} if necessary and run again.",
            )

        # Pipeline failures file
        pipeline_failures_csv = os.path.join(
            args.output_path,
            f"pipeline_failures_{args.pipeline_mode}_{current_time}.csv",
        )

        # Registration+vessel segmentation QC file
        qc_csv = os.path.join(
            args.output_path, f"qc_{args.pipeline_mode}_{current_time}.csv"
        )

        log_to_queue(log_queue, "Starting the template registration pipeline.")

        log_to_queue(log_queue, "Creating output directories in: " + args.output_path)
        analysisDir, inferenceDir, inferenceOutDir = setup_output_directories(
            args.output_path
        )

        if args.csv:
            log_to_queue(log_queue, "Processing data from: " + args.csv)
            df = process_csv(args.csv)
        elif args.directory:
            log_to_queue(log_queue, "Processing data from: " + args.directory)
            df = gather_nifti_data(
                args.directory
            )  # process_data(args.directory, args.output_path)
            log_to_queue(log_queue, f"Found {len(df)} files.")

        log_to_queue(log_queue, "Updating input dataframe.")
        df = update_input_dataframe_fields(
            df,
            args.output_path,
            analysisDir,
            inferenceDir,
            inferenceOutDir,
            registration_settings_path_local,
            args.pipeline_mode,
            log_queue,
        )

        num_threads = min(args.threads, len(df))
        runPipeline(
            df,
            args.output_path,
            num_threads,
            args.num_gpus,
            log_queue,
            pipeline_failures_csv,
            qc_csv,
            args.pipeline_mode,
            args.sliding_window,
        )

    except Exception as e:
        log_to_queue(log_queue, f"An error occurred: {e}", level=logging.ERROR)
    finally:
        log_queue.put(None)
        listener.join()


if __name__ == "__main__":
    main()
