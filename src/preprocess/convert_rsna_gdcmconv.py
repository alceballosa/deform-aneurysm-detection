import os
import subprocess
from pathlib import Path
from pdb import run

from joblib import Parallel, delayed


def rungdcmconv(input_folder, output_folder):
    for file in input_folder.glob("*.dcm"):
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

path_src = Path("/projects/vig/Datasets/aneurysm/competition/series/")
folders = sorted(list(path_src.glob("*")))
path_tgt = Path("/scratch/ceballosarroyo.a/aneurysm/mm_datasets/rsna/gdcmconv")
os.makedirs(path_tgt, exist_ok=True)


threads = 48
executor = Parallel(
    n_jobs=threads, backend="multiprocessing", prefer="processes", verbose=1
)
do = delayed(rungdcmconv)
tasks = (do( folder, path_tgt) for folder in folders)
executor(tasks)


