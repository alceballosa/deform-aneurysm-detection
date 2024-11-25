import copy
import time

import numpy as np
import SimpleITK as sitk
import torch
import torchvision
from detectron2.utils.registry import Registry
from src import transform

from .crop import InstanceCrop
from .crop2 import InstanceCrop2
from .split_comb import SplitComb
import edt 
DATA_MAPPER_REGISTRY = Registry("DATA_MAPPER")


@DATA_MAPPER_REGISTRY.register()
class CTADatasetMapper:
    LESION_LABELS = ["aneurysm"]

    def __init__(self, cfg, mode):
        """
        "no_transform" mode is used for TTA
        where images transformation is responsibilty of the model
        """
        assert mode in ["train", "val"]
        self.cfg = cfg
        if mode == "train":
            self.augmentations = self.build_transforms()
            self.crop_fn = self.build_crop_fn()
        else:
            self.split_comb = self.build_split_comb()
        self.mode = mode

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

    def __call__(self, dataset_dict):
        # start = time.perf_counter()
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
                for k in sample.keys():
                    if isinstance(sample[k], np.ndarray):
                        sample[k] = torch.tensor(sample[k])

                random_samples.append(sample)

            dataset_dict["samples"] = random_samples
        else:
            #     patches, nzhw = self.split_comb.split(data.pop("image"))
            #     dataset_dict["patches"] = torch.tensor(np.concatenate(patches, axis=0))
            #     dataset_dict["nzhw"] = nzhw
            dataset_dict["image"] = torch.tensor(data["image"])
            dataset_dict["image_spacing"] = data["image_spacing"]
            if self.cfg.MODEL.USE_VESSEL_INFO != "no":
                dataset_dict["mask"] = torch.tensor(data["mask"])
                
            if self.cfg.MODEL.USE_CVS_INFO != "no":
                dataset_dict["cvs_mask"] = torch.tensor(data["cvs_mask"])
        # end = time.perf_counter()
        # print("data processing and augmentation", end - end_loading)
        # print("data loading", end_loading - start)
        # print(
        #     f"input size {data['image'].shape} preprocess {time.perf_counter() - end_loading:.2f}  data loading {end_loading-start:.2f}",
        # )
        return dataset_dict

    def get_distance_map(self, mask):

        distances = edt.sdf(mask, black_border=False, parallel = 8)
        # distances[distances > 0] = 0
        return distances

    def load_data(self, dataset_dict):
        outputs = {}  
        image = sitk.ReadImage(dataset_dict["file_name"])
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
            lesion_index = np.sum(
                [csv_label[:, -1] == label for label in self.LESION_LABELS],
                axis=0,
                dtype="bool",
            )
            all_cls = np.ones(shape=(all_loc.shape[0]), dtype="int8") * (-1)
            # TODO: roll back
            all_cls[lesion_index] = 0
            outputs["all_loc"] = all_loc
            outputs["all_rad"] = all_rad
            outputs["all_cls"] = all_cls

        if self.cfg.MODEL.USE_VESSEL_INFO == "no":
            return outputs
        else:
            vessel_header = sitk.ReadImage(dataset_dict["vessel_file_name"])
            vessel = sitk.GetArrayFromImage(vessel_header).astype("float32")
            outputs["mask"] = vessel
            if self.cfg.MODEL.USE_CVS_INFO != "no":
                cvs_header = sitk.ReadImage(dataset_dict["cvs_file_name"])
                cvs = sitk.GetArrayFromImage(cvs_header).astype("float32")
                # cvs = self.get_distance_map(cvs)
                outputs["cvs_mask"] = cvs
            # vessel = self.get_distance_map(sitk.GetArrayFromImage(vessel))
            return outputs

    def normalize(self, data):
        min_value, max_value = self.cfg.DATA.WINDOW
        data[data > max_value] = max_value
        data[data < min_value] = min_value
        data = (data - min_value) / (max_value - min_value)
        return data


def sphere(r=1):
    x = np.arange(2 * r + 1).reshape(1, 1, -1) - r
    y = np.arange(2 * r + 1).reshape(1, -1, 1) - r
    z = np.arange(2 * r + 1).reshape(-1, 1, 1) - r
    distance = np.sqrt(x * x + y * y + z * z)
    return distance <= r
