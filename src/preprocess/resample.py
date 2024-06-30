import SimpleITK as sitk
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed


def resample_simg(itkimage:sitk.Image, newSpacing=(1.0, 1.0, 1.0), label=False)->sitk.Image:
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSpacing:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    newSpacing = np.array(newSpacing, float)
    originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSpacing / originSpcaing
    newSize = originSize / factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if label:
        resamplemethod=sitk.sitkNearestNeighbor
    else:
        resamplemethod=sitk.sitkLinear
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

target_spacing = np.array((0.4, 0.4, 0.4))
im_dir = "/work/vig/Datasets/aneurysm/internal_train"
mask_dir = "/work/vig/Datasets/aneurysm/internal_label_train"

def run(im_f, mask_f):
    image = sitk.ReadImage(im_f)
    mask = sitk.ReadImage(mask_f)
    space = np.array(image.GetSpacing())
    if np.abs(space-target_spacing).max() < 0.01:
        print(space)
        # sitk.WriteImage(image, im_f.replace("internal_train", "internal_train_0.4"))
        # sitk.WriteImage(mask,  mask_f.replace("internal_label_train", "internal_label_train_0.4"))
        sitk.WriteImage(image, im_f.replace("internal_test", "internal_test_0.4"))
        sitk.WriteImage(mask,  mask_f.replace("internal_label_test", "internal_label_test_0.4"))
    else:
        print('resample before', space, image.GetSize())
        image_resampled = resample_simg(itkimage=image, newSpacing=target_spacing, label=False)
        mask_resampled = resample_simg(itkimage=mask, newSpacing=target_spacing, label=True)
        new_spacing = image_resampled.GetSpacing()
        assert image_resampled.GetSize() == mask_resampled.GetSize()
        print('resample after', new_spacing, image_resampled.GetSize())
        # sitk.WriteImage(image_resampled, im_f.replace("internal_train", "internal_train_0.4"))
        # sitk.WriteImage(mask_resampled,  mask_f.replace("internal_label_train", "internal_label_train_0.4"))
        sitk.WriteImage(image_resampled, im_f.replace("internal_test", "internal_test_0.4"))
        sitk.WriteImage(mask_resampled,  mask_f.replace("internal_label_test", "internal_label_test_0.4"))

if __name__ == "__main__":
    target_spacing = np.array((0.4, 0.4, 0.4))
    # im_dir = "/work/vig/Datasets/aneurysm/internal_train"
    # mask_dir = "/work/vig/Datasets/aneurysm/internal_label_train"
    
    im_dir = "/work/vig/Datasets/aneurysm/internal_test"
    mask_dir = "/work/vig/Datasets/aneurysm/internal_label_test"
    
    im_files = sorted(list(glob(f"{im_dir}/*")))
    mask_files = sorted(list(glob(f"{mask_dir}/*")))
    
    num_workers = 8
    executor = Parallel(n_jobs=num_workers, backend="multiprocessing", prefer="processes", verbose=1)
    do = delayed(run)
    tasks = (do(im_f, mask_f) for im_f, mask_f in zip(im_files, mask_files))
    executor(tasks)