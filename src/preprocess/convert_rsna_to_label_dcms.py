import os
import subprocess
from pathlib import Path
from pdb import run

import numpy as np
import pandas as pd
import pydicom
from joblib import Parallel, delayed


def rungdcmconv(input_folder, output_folder, df_localizers):
    for file in input_folder.glob("*.dcm"):
        path_new = output_folder / file.parent.name / file.name
        os.makedirs(path_new.parent, exist_ok=True)
        # load the dicom file
        dicom = pydicom.dcmread(str(file))
        # create a dicom with the exact same metadata but with an empty pixel array
        zeros = np.zeros(dicom.pixel_array.shape, dtype=dicom.pixel_array.dtype)
        # get the dicom array
        # get the localizer row
        rows = df_localizers[df_localizers["SeriesInstanceUID"] == file.parent.name]

        if rows.shape[0] > 0:
            for _, row in rows.iterrows():
                instanceuid = row["SOPInstanceUID"]
                if instanceuid != dicom.SOPInstanceUID:
                    continue
                coord = eval(row["coordinates"])
                print(coord)
                if len(coord.keys()) == 2:
                    x, y = int(coord["x"]), int(coord["y"])
                    zeros[y-5:y+5, x-5:x+5] = 1000
                elif len(coord.keys()) == 3:
                    x, y, z = int(coord["x"]), int(coord["y"]), int(coord["f"])
                    zeros[z, y-5:y+5, x-5:x+5] = 1000
            
        dicom.set_pixel_data(
            zeros, photometric_interpretation="MONOCHROME2", bits_stored=16
        )
        #print(dicom.pixel_array)

        dicom.save_as(str(path_new))


path_localizers = "/projects/vig/Datasets/aneurysm/competition/train_localizers.csv"
df_localizers = pd.read_csv(path_localizers)
path_src = Path("/scratch/ceballosarroyo.a/aneurysm/mm_datasets/rsna/gdcmconv")
folders = sorted(list(path_src.glob("*")))


path_tgt = Path("/scratch/ceballosarroyo.a/aneurysm/mm_datasets/rsna/label_dcms")
os.makedirs(path_tgt, exist_ok=True)

print(len(folders))

threads = 48
executor = Parallel(
    n_jobs=threads, backend="multiprocessing", prefer="processes", verbose=1
)
do = delayed(rungdcmconv)
tasks = (do(folder, path_tgt, df_localizers) for folder in folders)
executor(tasks)
