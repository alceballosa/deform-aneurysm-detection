import os
import subprocess
import sys
from pathlib import Path
from pdb import run

import dicom2nifti
from joblib import Parallel, delayed


def run_dcm2niix(path_executable, input_folder, output_folder):
    subprocess.run(
        [path_executable, "-z", "y",
                    "-f", "%f",
                    "-b", "y", 
                    "-o", str(output_folder), 
                    "-i", "n",
                    str(input_folder)]
    )

path_executable = "/projects/vig/alberto/medical/dcm2niix"
path_src = Path("/scratch/ceballosarroyo.a/aneurysm/mm_datasets/rsna/gdcmconv")
folders = list(path_src.glob("*"))
path_tgt = Path("/scratch/ceballosarroyo.a/aneurysm/mm_datasets/rsna/nifti")
os.makedirs(path_tgt, exist_ok=True)


threads = 48
executor = Parallel(
    n_jobs=threads, backend="multiprocessing", prefer="processes", verbose=1
)
do = delayed(run_dcm2niix)
tasks = (do(path_executable, folder, path_tgt) for folder in folders)
executor(tasks)


