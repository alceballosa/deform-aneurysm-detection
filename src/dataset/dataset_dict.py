import os

import numpy as np
import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.registry import Registry

DATASET_FUNC_REGISTRY = Registry("SEG_DATASET_FUNCTION")


def setup_data_catalog(cfg):
    # id_map = {i: i for i in range(len(ROOM_TYPE_NAMES))}
    DatasetFunction = DATASET_FUNC_REGISTRY.get(cfg.CUSTOM.DATASET_FUNCTION)

    def register_dataset(name, mode):
        dataset_fn = DatasetFunction(cfg, mode=mode)
        DatasetCatalog.register(name, dataset_fn)
        MetadataCatalog.get(name).set(
            stuff_classes=["aneurysm"],
        )

    datasets = set(cfg.DATASETS.TRAIN + cfg.DATASETS.TEST)
    for dataset in datasets:
        mode = dataset.split("_")[-1]
        register_dataset(dataset, mode)


@DATASET_FUNC_REGISTRY.register()
class CTADatasetFunction:
    def __init__(self, cfg, mode="train"):
        assert mode in ["train", "val"]
        self.cfg = cfg
        self.mode = mode

    def __call__(self):
        data_dir_cfg = {
            "train": self.cfg.DATA.DIR.TRAIN,
            "val": self.cfg.DATA.DIR.VAL,
        }[self.mode]
        # NOTE: here is where we can reduce dataset size for debugging
        scan_ids = sorted(os.listdir(data_dir_cfg.SCAN_DIR))  # [6:7]
        if self.cfg.MODEL.EVAL_VIZ_MODE:
            print("MODO EVALLL")
            scan_ids = scan_ids[4:12]

        annotations = (
            np.array(
                pd.read_csv(data_dir_cfg.ANNOTATION_FILE)[
                    ["seriesuid", "coordX", "coordY", "coordZ", "w", "h", "d", "lesion"]
                ]
            )
            if self.mode == "train"
            else None
        )

        dataset_dicts = []
        for scan_id in scan_ids:
            record = {}
            record["scan_id"] = scan_id
            record["file_name"] = os.path.join(data_dir_cfg.SCAN_DIR, scan_id)
            record["annotations"] = (
                annotations[annotations[:, 0] == scan_id, 1:]
                if self.mode == "train"
                else None
            )
            if self.cfg.MODEL.USE_VESSEL_INFO != "no":
                record["vessel_file_name"] = os.path.join(
                    data_dir_cfg.VESSEL_DIR, scan_id
                )
            if self.cfg.MODEL.USE_CVS_INFO != "no":
                record["cvs_file_name"] = os.path.join(data_dir_cfg.CVS_DIR, scan_id)

            dataset_dicts.append(record)
        if self.cfg.CUSTOM.DEBUG:
            return dataset_dicts[: self.cfg.CUSTOM.DEBUG_DATASET_SIZE]
        return dataset_dicts


@DATASET_FUNC_REGISTRY.register()
class DUMMY:
    def __init__(self, cfg, mode="train"):
        assert mode in ["train", "val"]
        self.cfg = cfg
        self.mode = mode

    def __call__(self):
        return [{"scan_id": i for i in range(self.cfg.CUSTOM.DUMMY_SIZE)}]
