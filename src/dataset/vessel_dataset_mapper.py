import copy
import time

import numpy as np
import SimpleITK as sitk
import torch
import torchvision

from src import transform

from .crop2 import SPInstanceCrop2
from .dataset_mapper import DATA_MAPPER_REGISTRY, CTADatasetMapper
from .vessel_crop import SPInstanceCrop


@DATA_MAPPER_REGISTRY.register()
class CTAVesselDatasetMapper(CTADatasetMapper):
    def build_crop_fn(self):
        cfg = self._load_crop_cfg()
        # return SPInstanceCrop(**cfg)
        return SPInstanceCrop2(**cfg)

    def build_transforms(self):
        crop_size = self.cfg.DATA.PATCH_SIZE
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

    def load_data(self, dataset_dict):
        #!start = time.perf_counter()
        data = super().load_data(dataset_dict)
        #!cached = False
        if self.cfg.CUSTOM.CACHE and dataset_dict["vessel_file_name"] in self.cache:
            vessel = self.cache[dataset_dict["vessel_file_name"]].copy()
            #!cached = True
        else:
            vessel = sitk.ReadImage(dataset_dict["vessel_file_name"])
            vessel = sitk.GetArrayFromImage(vessel)

            if self.cfg.CUSTOM.CACHE:
                self.cache[dataset_dict["vessel_file_name"]] = vessel.copy()

        data["mask"] = vessel
        #! print(f"cached {cached} times ", time.perf_counter() - start)
        return data

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
                for k in sample.keys():
                    if isinstance(sample[k], np.ndarray):
                        sample[k] = torch.tensor(sample[k])

                #! debug with full mask
                if self.cfg.CUSTOM.DEBUG and self.cfg.CUSTOM.FULL_MASK:
                    sample["mask"] = torch.ones_like(sample["image"]).bool()
                random_samples.append(sample)

            dataset_dict["samples"] = random_samples
        else:
            dataset_dict["image"] = torch.tensor(data["image"])
            dataset_dict["mask"] = torch.tensor(data["mask"])
            #! debug with full mask
            if self.cfg.CUSTOM.DEBUG and self.cfg.CUSTOM.FULL_MASK:
                dataset_dict["mask"] = torch.ones_like(dataset_dict["image"]).bool()
            dataset_dict["image_spacing"] = data["image_spacing"]
        # end = time.perf_counter()
        # print("data processing and augmentation", end - end_loading)
        # print("data loading", end_loading - start)
        return dataset_dict
