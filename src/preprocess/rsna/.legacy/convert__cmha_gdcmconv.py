import os
import subprocess
from pathlib import Path
from pdb import run

from joblib import Parallel, delayed


def rungdcmconv(input_folder, output_folder):
    for file in input_folder.glob("*.DCM"):
        path_new = output_folder / file.parent.name / file.name
        os.makedirs(path_new.parent, exist_ok=True)
        subprocess.run(
            [
                "gdcmconv",
                "--raw",
                str(file),
                str(path_new)
            ]
        )

path_src = Path("/projects/vig/Datasets/aneurysm/cta_datasets/cmha_review/dicom/")
folders = sorted(list(path_src.glob("*")))
print(folders)
path_tgt = Path("/projects/vig/Datasets/aneurysm/cta_datasets/cmha_review/dicom_conv")
os.makedirs(path_tgt, exist_ok=True)


threads = 1
executor = Parallel(
    n_jobs=threads, backend="multiprocessing", prefer="processes", verbose=1
)
do = delayed(rungdcmconv)
tasks = (do( folder/"CTA images", path_tgt/folder.name ) for folder in folders)
executor(tasks)


