"""
Resample images to 0.4mm spacing. The target dir will be
the same as the input dir with "_0.4" appended to the name.
"""

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from joblib import Parallel, delayed

target_spacing = np.array((0.4, 0.4, 0.4))


def resample(
    itkimage: sitk.Image, newSpacing=(1.0, 1.0, 1.0), label=False
) -> sitk.Image:
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSpacing:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    newSpacing = np.array(newSpacing, float)
    originalSpacing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSpacing / originalSpacing
    newSize = originSize / factor
    newSize = newSize.astype(int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if label:
        resamplemethod = sitk.sitkNearestNeighbor
    else:
        resamplemethod = sitk.sitkLinear
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled


def fix_case(im_sitk, mask_sitk, name):
    """
    Manually fixes some cases where masks and images have different sizes
    resulting in misalignment.
    """
    if name == "ExtA0009.nii.gz":
        print("Fixing case", name)
        im_np = sitk.GetArrayFromImage(im_sitk)
        im_np = im_np[10:-5, :, :]
        im_sitk = sitk.GetImageFromArray(im_np)
        im_sitk.CopyInformation(mask_sitk)
    elif name == "ExtA0032.nii.gz":
        print("Fixing case", name)
        im_np = sitk.GetArrayFromImage(im_sitk)
        mask_np = sitk.GetArrayFromImage(mask_sitk)
        mask_np = mask_np[:, 49:-49, 49:-49]
        mask_sitk = sitk.GetImageFromArray(mask_np)
        mask_sitk.CopyInformation(im_sitk)
    return im_sitk, mask_sitk


def process(path_im, path_mask):
    image = sitk.ReadImage(path_im)
    tgt_path_im = Path(str(path_im.parent) + "_0.4") / path_im.name
    if path_mask != "":
        mask = sitk.ReadImage(path_mask)
        tgt_path_mask = Path(str(path_mask.parent) + "_0.4") / path_mask.name
    else:
        mask = None
    space = np.array(image.GetSpacing())
    if space.shape == (4,):
        print("\n4D image, skipping:", path_im)
        return

    if np.abs(space[0:3] - target_spacing[0:3]).max() < 0.01:
        print("Maintained original spacing:", space)
        sitk.WriteImage(image, str(tgt_path_im))
        if path_mask:
            sitk.WriteImage(mask, str(tgt_path_mask))
    else:
        # optionally fix certain cases where mask and image may not match
        image, mask = fix_case(image, mask, path_im.name)
        image_resampled = resample(
            itkimage=image, newSpacing=target_spacing, label=False
        )
        shape = np.array(image_resampled.GetSize())
        if shape[2] == 0:
            print("\nEmpty image after resample, skipping:", path_im)
            return
        if path_mask != "":
            mask_resampled = resample(
                itkimage=mask, newSpacing=target_spacing, label=True
            )
            print(path_im, image.GetSize(), path_mask, mask.GetSize())
            assert image_resampled.GetSize() == mask_resampled.GetSize()
        new_spacing = image_resampled.GetSpacing()

        print("Spacing/size before and after:", space, image.GetSize(), new_spacing, image_resampled.GetSize())

        sitk.WriteImage(image_resampled, tgt_path_im)
        if path_mask:
            sitk.WriteImage(mask_resampled, tgt_path_mask)


if __name__ == "__main__":
    im_dir = Path(sys.argv[1])
    tgt_im_dir = im_dir.parent / (im_dir.name + "_0.4")
    tgt_im_dir.mkdir(exist_ok=True, parents=True)
    try:
        mask_dir = Path(sys.argv[2])
        tgt_mask_dir = mask_dir.parent / (mask_dir.name + "_0.4")
        tgt_mask_dir.mkdir(exist_ok=True, parents=True)
    except IndexError:
        mask_dir = ""

    im_files = sorted(list(im_dir.glob("*")))

    if mask_dir == "":
        mask_files = [""] * len(im_files)
    else:
        mask_files = sorted(list(mask_dir.glob("*")))
    num_workers = 2 # 4 if len(im_files) > 4 else len(im_files)
    executor = Parallel(
        n_jobs=num_workers, backend="multiprocessing", prefer="processes", verbose=2
    )
    do = delayed(process)
    tasks = (do(im_f, mask_f) for im_f, mask_f in zip(im_files, mask_files))
    executor(tasks)
