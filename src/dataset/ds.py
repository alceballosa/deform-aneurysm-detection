from torch.utils.data import Dataset
from src import transform
import torch 
import torchvision
import numpy as np
import pandas as pd 
import os
from pathlib import Path 
from .crop import InstanceCrop
from .crop2 import InstanceCrop2
from .split_comb import SplitComb
import SimpleITK as sitk
import copy

def resolve_path(path):
    return str(Path(path).resolve())

def maybe_read_from_ram(file_name):
    """
    Tries to read a file from RAM (specifically /dev/shm), if not
    defaults to OG location.
    """
    new_folder = "/dev/shm/"
    sample_name = "/".join(file_name.split("/")[-3:])
    new_file_name = os.path.join(new_folder, sample_name)
    try:
        size_og = os.path.getsize(file_name)
        size_new = os.path.getsize(new_file_name)
        if size_og == size_new:
            # print(f"Reading from RAM: {new_file_name}")
            return sitk.ReadImage(new_file_name)
        else:
            print(
                f"Sz mismatch: {file_name} ({size_og}) != {new_file_name} ({size_new})"
            )
            return sitk.ReadImage(file_name)
    except:
        # print(
        #    f"File not found in RAM: {new_file_name}, reading from original location."
        # )
        return sitk.ReadImage(file_name)


class CTA_Dataset(Dataset):
    LESION_LABELS = ["aneurysm", "non_aneurysm"]
    LESION_IDS = {
        "aneurysm": 0,  # aneurysm
        "non_aneurysm": 1,  # non_aneurysm
    }    
    def __init__(self, cfg, mode):
        super().__init__()    
        assert mode in ["train", "val"]
        self.cfg = cfg
        if mode == "train":
            self.augmentations = self.build_transforms()
            self.crop_fn = self.build_crop_fn()
        else:
            self.split_comb = self.build_split_comb()
        self.mode = mode
        data_dir_cfg = {
            "train": self.cfg.DATA.DIR.TRAIN,
            "val": self.cfg.DATA.DIR.VAL,
        }[self.mode]
        scan_ids = sorted(os.listdir(data_dir_cfg.SCAN_DIR))
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
            record["file_name"] = resolve_path(os.path.join(data_dir_cfg.SCAN_DIR, scan_id))
            record["annotations"] = (
                annotations[annotations[:, 0] == scan_id, 1:]
                if self.mode == "train"
                else None
            )
            if self.cfg.MODEL.USE_VESSEL_INFO != "no":
                record["vessel_file_name"] = resolve_path(os.path.join(
                    data_dir_cfg.VESSEL_DIR, scan_id
                ))
            if self.cfg.MODEL.USE_CVS_INFO != "no":
                record["cvs_file_name"] = resolve_path(os.path.join(data_dir_cfg.CVS_DIR, scan_id))
            
            dataset_dicts.append(record)
        if self.cfg.CUSTOM.DEBUG:
            return dataset_dicts[: self.cfg.CUSTOM.DEBUG_DATASET_SIZE]
        self.dataset_dicts = dataset_dicts

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        dataset_dict = self.dataset_dicts[idx]

        dataset_dict = copy.deepcopy(dataset_dict)
        data = self.load_data(dataset_dict)
        # end_loading = time.perf_counter()
        if self.mode == "train":
            samples = self.crop_fn(data)
            random_samples = []
            for sample in samples:
                if self.augmentations:
                    sample = self.augmentations(sample)
                # sample["image"] = sample["image"] * 2.0 - 1.0  # normalized to -1 ~ 1
                # TODO: review this
                # for k in sample.keys():
                #     if isinstance(sample[k], np.ndarray):
                #         sample[k] = torch.tensor(sample[k])

                random_samples.append(sample)

            dataset_dict["samples"] = random_samples
        else:
            #     patches, nzhw = self.split_comb.split(data.pop("image"))
            #     dataset_dict["patches"] = torch.tensor(np.concatenate(patches, axis=0))
            #     dataset_dict["nzhw"] = nzhw
            dataset_dict["image"] = torch.tensor(data["image"], device="cpu")
            dataset_dict["image_spacing"] = data["image_spacing"]
            if self.cfg.MODEL.USE_VESSEL_INFO != "no":
                dataset_dict["mask"] = torch.tensor(data["mask"], device="cpu")

            if self.cfg.MODEL.USE_CVS_INFO != "no":
                dataset_dict["cvs_mask"] = torch.tensor(data["cvs_mask"], device="cpu")
        # end = time.perf_counter()
        # print("data processing and augmentation", end - end_loading)
        # print("data loading", end_loading - start)
        # print(
        #     f"input size {data['image'].shape} preprocess {time.perf_counter() - end_loading:.2f}  data loading {end_loading-start:.2f}",
        # )
        return dataset_dict

    def load_data(self, dataset_dict):
        outputs = {}
        # image = sitk.ReadImage(dataset_dict["file_name"])
        image = maybe_read_from_ram(dataset_dict["file_name"])
        image_spacing = image.GetSpacing()[::-1]  # z, y, x
        image = sitk.GetArrayFromImage(image).astype("float32")  # z, y, x
        # NOTE: normalize on gpu is faster
        # image = self.normalize(image)  # normalized
        outputs["image"] = image
        outputs["image_spacing"] = image_spacing
        outputs["scan_id"] = dataset_dict["scan_id"]

        if self.mode == "train":
            csv_label = dataset_dict["annotations"]
            all_loc = csv_label[:, 0:3].astype("float32")  # x,y,z
            all_loc = all_loc[:, ::-1]  # convert z,y,x
            all_rad = csv_label[:, 3:6].astype("float32")  # w,h,d
            all_rad = all_rad[:, ::-1]  # convert d,h,w
            all_cls_text = csv_label[:, -1]  # lesion type
            all_cls = np.array(
                [self.LESION_IDS[label] for label in all_cls_text], dtype="int8"
            )  # convert to int8
            # lesion_index = np.sum(
            #    [csv_label[:, -1] == label for label in self.LESION_LABELS],
            #    axis=0,
            #    dtype="bool",
            # )
            # all_cls = np.zeros(shape=(all_loc.shape[0]), dtype="int8") # * (-1)
            # TODO: roll back
            # all_cls[lesion_index] = 0

            outputs["all_loc"] = all_loc
            outputs["all_rad"] = all_rad
            outputs["all_cls"] = all_cls

        if self.cfg.MODEL.USE_VESSEL_INFO == "no":
            return outputs
        else:

            # vessel_header = sitk.ReadImage(dataset_dict["vessel_file_name"])
            vessel_header = maybe_read_from_ram(dataset_dict["vessel_file_name"])
            vessel = sitk.GetArrayFromImage(vessel_header).astype("float32")
            outputs["mask"] = vessel
            if self.cfg.MODEL.USE_CVS_INFO != "no":
                # cvs_header = sitk.ReadImage(dataset_dict["cvs_file_name"])
                cvs_header = maybe_read_from_ram(dataset_dict["cvs_file_name"])
                cvs = sitk.GetArrayFromImage(cvs_header).astype("float32")
                # cvs = self.get_distance_map(cvs)
                outputs["cvs_mask"] = cvs
            # vessel = self.get_distance_map(sitk.GetArrayFromImage(vessel))
            return outputs

    def build_crop_fn(self):
        cfg = self._load_crop_cfg()
        # return InstanceCrop(**cfg)
        return InstanceCrop2(**cfg)

    def _load_crop_cfg(self):
        cfg = self.cfg

        random_space = cfg.DATA.CROPPING_AUG.SPACING
        if (random_space[1] - random_space[0]) < 1e-3:
            random_space = None

        random_rotation = cfg.DATA.CROPPING_AUG.ROTATION
        if max(random_rotation) < 1e-3:
            random_rotation = None

        random_translation = cfg.DATA.CROPPING_AUG.TRANSLATION
        if max(random_translation) < 1e-3:
            random_translation = None

        return dict(
            crop_size=cfg.DATA.PATCH_SIZE,
            rand_trans=random_translation,
            rand_rot=random_rotation,
            rand_space=random_space,
            spacing=cfg.DATA.SPACING,
            overlap=cfg.DATA.OVERLAP,
            tp_ratio=cfg.DATA.CROPPING_AUG.TP_RATIO,
            sample_num=cfg.SOLVER.SAMPLES_PER_SCAN,
            blank_side=cfg.DATA.CROPPING_AUG.BLANK_SIDE,
        )

    def build_split_comb(self):
        cfg = self.cfg
        return SplitComb(
            crop_size=cfg.DATA.PATCH_SIZE,
            overlap=cfg.DATA.OVERLAP,
            pad_value=cfg.DATA.WINDOW[0],  # padding min value of window
        )

    def build_transforms(self):
        crop_size = self.cfg.DATA.PATCH_SIZE
        if self.cfg.MODEL.USE_VESSEL_INFO == "no":
            transform_list_train = [
                transform.RandomFlip(
                    flip_depth=True, flip_height=True, flip_width=True, p=0.5
                ),
                transform.RandomTranspose(
                    p=0.5, trans_xy=True, trans_zx=False, trans_zy=False
                ),
                transform.Pad(output_size=crop_size),
                transform.RandomCrop(output_size=crop_size, pos_ratio=0.9),
                transform.CoordToAnnot(),
            ]
            train_transform = torchvision.transforms.Compose(transform_list_train)

            return train_transform
        else:
            transform_list_train = [
                transform.RandomMaskFlip(
                    flip_depth=True, flip_height=True, flip_width=True, p=0.5
                ),
                transform.RandomMaskTranspose(
                    p=0.5, trans_xy=True, trans_zx=False, trans_zy=False
                ),
                transform.MaskPad(output_size=crop_size),
                transform.RandomMaskCrop(output_size=crop_size, pos_ratio=0.9),
                transform.CoordToAnnot(),
            ]
            train_transform = torchvision.transforms.Compose(transform_list_train)
            return train_transform