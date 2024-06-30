import os

import numpy as np
import pandas as pd

from .dataset_dict import DATASET_FUNC_REGISTRY, CTADatasetFunction


@DATASET_FUNC_REGISTRY.register()
class CTAVesselDatasetFunction(CTADatasetFunction):
    """
    also use vessel segmentation mass
    """

    def __call__(self):
        data_dir_cfg = {
            "train": self.cfg.DATA.DIR.TRAIN,
            "val": self.cfg.DATA.DIR.VAL,
        }[self.mode]

        scan_ids = sorted(os.listdir(data_dir_cfg.SCAN_DIR))
        annotations = np.array(
            pd.read_csv(data_dir_cfg.ANNOTATION_FILE)[
                ["seriesuid", "coordX", "coordY", "coordZ", "w", "h", "d", "lesion"]
            ]
        )

        dataset_dicts = []
        for scan_id in scan_ids:
            record = {}
            record["scan_id"] = scan_id
            record["file_name"] = os.path.join(data_dir_cfg.SCAN_DIR, scan_id)
            record["vessel_file_name"] = os.path.join(data_dir_cfg.VESSEL_DIR, scan_id)
            record["annotations"] = annotations[annotations[:, 0] == scan_id, 1:]
            dataset_dicts.append(record)
        if self.cfg.CUSTOM.DEBUG and self.cfg.CUSTOM.DEBUG_DATASET_SIZE:
            return dataset_dicts[: self.cfg.CUSTOM.DEBUG_DATASET_SIZE]
        return dataset_dicts
