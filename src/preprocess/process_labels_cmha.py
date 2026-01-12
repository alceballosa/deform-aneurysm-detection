import os
from pathlib import Path

import slicer

# path_files = Path("/Users/alberto/Downloads/26965450/patients/")
path = Path(
    r"C:\\Users\\alber\\Downloads\\aneurysm_thesis\\fixed_files_cmha\\AHMU1218077"
)
stl_file_name = path / "aneurysm_AHMU1218077_1.stl"
reference_volume_path = Path(
    r"C:\\Users\\alber\\Downloads\\aneurysm_thesis\\fixed_files_cmha\\077.nii.gz"
)

output_file_name = str(path / f"{reference_volume_path.name}_label.nii.gz")
referenceVolumeNode = slicer.util.loadVolume(reference_volume_path)
segmentationNode = slicer.util.loadSegmentation(stl_file_name)  ## stl
stl_file_name = path / "aneurysm_AHMU1218077_2.stl"
segmentationNode2 = slicer.util.loadSegmentation(stl_file_name)  ## stl

outputLabelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass(
    "vtkMRMLLabelMapVolumeNode"
)
slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
    segmentationNode, outputLabelmapVolumeNode, referenceVolumeNode
)
slicer.util.saveNode(outputLabelmapVolumeNode, output_file_name)
# close files
slicer.mrmlScene.Clear(0)


# stl_file_name = "/Users/alberto/Downloads/26965450/patients/AHMU1218001/3D_aneurysm_AHMU1218001.stl"
# output_file_name = "/Users/alberto/Downloads/26965450/patients/AHMU1218001/3D_aneurysm_AHMU1218001.nii.gz"
# reference_volume_path = "/Users/alberto/Downloads/26965450/patients/AHMU1218001/cta_images_head_AHMU1218001.nii.gz"
