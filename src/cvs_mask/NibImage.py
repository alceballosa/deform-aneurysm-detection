

import nibabel as nib
import numpy as np
import ants

class NibImage():
    def __init__(self, img_path):
        self.path = img_path
        self.img = nib.load(img_path)
        
    def crop_cta_volume(self, crop_height=170, save=False, save_path=None):
        self.crop_height = crop_height
        img_data = self.img.get_fdata()


        voxel_size = self.img.header.get_zooms()[2]
        crop_slices = int(crop_height / voxel_size)

        if img_data.shape[2] <= crop_slices:
            return self.img
            # If the volume is less than or equal to crop height, just copy the images
            #shutil.copy(cta_scan_path, os.path.join(output_dir, os.path.basename(cta_scan_path)))
            #return os.path.join(output_dir, os.path.basename(cta_scan_path))

        cropped_data = img_data[:, :, -crop_slices:]
        cropped_img = nib.Nifti1Image(cropped_data, self.img.affine, self.img.header)
        if save:
            nib.save(cropped_img, save_path)

        return cropped_img
    
    def reverse_crop_cta(self, cropped_data, save=False, save_path=None):
        #def merge_with_zeros(cta_scan_path, output_dir, total_slices):
        total_slices = self.img.shape[2]
        img_data = cropped_data

        # Determine the number of slices to pad with zeros
        num_slices_to_pad = total_slices - img_data.shape[2]
        print('num_slices_to_pad:', num_slices_to_pad, 'total_slices:', total_slices)

        if num_slices_to_pad > 0:
            zero_data = np.zeros((img_data.shape[0], img_data.shape[1], num_slices_to_pad))
            merged_data = np.zeros((img_data.shape[0], img_data.shape[1], total_slices))
            merged_data[:, :, :num_slices_to_pad] = zero_data
            merged_data[:, :, num_slices_to_pad:] = img_data
        else:
            merged_data = img_data

        # Debug statements
        original_bbox = np.nonzero(img_data)
        merged_bbox = np.nonzero(merged_data)

        #print(f"Original bounding box: min {np.min(original_bbox, axis=1)}, max {np.max(original_bbox, axis=1)}")
        #print(f"Merged bounding box: min {np.min(merged_bbox, axis=1)}, max {np.max(merged_bbox, axis=1)}")

        print('merge_data:', merged_data.shape)

        merged_img = nib.Nifti1Image(merged_data, self.img.affine, self.img.header)
        
        if save:
            nib.save(merged_img, save_path)

        return merged_img
    
    def find_cvs_bbox(self, data, save=False, save_path=None):

        indices = np.argwhere(data == 1)

        # Determine the bounding box
        bbox = np.zeros(data.shape)
        #try:
        x_min, y_min, z_min = np.min(indices, axis=0)
        x_max, y_max, z_max = np.max(indices, axis=0)

        
        bbox[x_min:x_max, y_min:y_max, z_min:z_max] = 1
        
        """except:
            print("Could not create bounding box")
            x=input("Type 'y' to save empty mask or anything else to abort run: ")
            if x == 'y':
                pass 
            else:
                raise ValueError("Could not create bounding box")
        """
        print(np.sum(bbox))
        bbox_img = nib.Nifti1Image(bbox, self.img.affine, self.img.header)
        if save:
            nib.save(bbox_img, save_path)

        return bbox_img


    def expand_bbox(self, bbox_data, expand_distance = [8,8,8], save=False, save_path=None):

        expand_bbox = np.zeros(bbox_data.shape)
        indices = np.argwhere(bbox_data == 1)
        x_shape, y_shape, z_shape = bbox_data.shape

        # Determine the bounding box
        x_min, y_min, z_min = np.min(indices, axis=0)
        x_max, y_max, z_max = np.max(indices, axis=0)

        voxel_size = self.img.header.get_zooms()

        x_slice, y_slice, z_slice = int(expand_distance[0]/voxel_size[0]), int(expand_distance[1]/voxel_size[1]), int(expand_distance[2]/voxel_size[2])

        x_range = [np.maximum(x_min-x_slice,0), np.minimum(x_max+x_slice,x_shape-1)]
        y_range = [np.maximum(y_min-y_slice,0), np.minimum(y_max+y_slice,y_shape-1)]
        z_range = [np.maximum(z_min-z_slice,0), np.minimum(z_max+z_slice,z_shape-1)]

        expand_bbox[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]] = 1
        
        bbox_img = nib.Nifti1Image(expand_bbox, self.img.affine, self.img.header)
        if save:
            nib.save(bbox_img, save_path)

        return bbox_img
    
    def clip_image(self, nib_data, save=False, save_path=None):
        clip_data = nib_data

        np.clip(clip_data, 0, np.max(clip_data), out=clip_data)

        clip_img = nib.Nifti1Image(clip_data, self.img.affine, self.img.header)

        if save:
            nib.save(clip_img, save_path)

        return clip_img

        #fixed_image_clip = ants.from_numpy(cropped_data, spacing=fixed_image.spacing, origin=fixed_image.origin, direction=fixed_image.direction)
        #ants.image_write(fixed_image_clip, './cropped_img.nii.gz')


    def overlay_vein_mask(self, vessel_img, bbox_img, save=False, save_path=None):

        #vessel_img = nib.load(vessel_seg_path)
        vessel_data = vessel_img.get_fdata()
        vessel_data[vessel_data!=2] = 0

        vein_data = vessel_data
        bbox_data = bbox_img.get_fdata()
        #print(vein_data.shape, bbox_data.shape)

        cvs_mask_data = np.logical_and(vein_data, bbox_data)
        mask_img = nib.Nifti1Image(cvs_mask_data, self.img.affine, self.img.header)

        if save:
            nib.save(mask_img, save_path)

        return mask_img

    def remove_vein_bbox(self, vessel_img, cvs_mask_img, aneurysm_bbox_img, save=False, save_path='./aneurysm_bbox.nii.gz'):
        #cvs_mask_img = nib.load('./cvs_mask.nii.gz')
        #aneurysm_img = nib.load(aneurysm_bbox_path)
        #vessel_img = nib.load(vessel_seg_path)

        aneurysm_data = aneurysm_bbox_img.get_fdata()
        aneurysm_data[aneurysm_data>0] =1

        cvs_mask_data = cvs_mask_img.get_fdata()
        vessel_data = vessel_img.get_fdata()
        vessel_data[vessel_data!=2] = 0
        vein_data = vessel_data

        vein_wo_cvs = np.copy(vein_data)
        vein_wo_cvs[cvs_mask_data==1] = 0

        overlay = np.logical_and(vein_wo_cvs, aneurysm_data)
        aneurysm_bbox = np.zeros(aneurysm_data.shape)
        for z in range(overlay.shape[2]):
            overlay_slice = overlay[:,:,z]
            
            aneurysm_slice = aneurysm_data[:,:,z]
            if np.sum(overlay_slice)>0:
                
                #indices = np.argwhere(slice == 1)
                count, id = num_of_boungingbox(overlay_slice)
                for c in range(count): 
                    x, y = id[c] 
                    remove_boundingbox(aneurysm_slice, x, y)

            aneurysm_bbox[:,:, z] = aneurysm_slice

        bbox_img = nib.Nifti1Image(aneurysm_bbox, self.img.affine, self.img.header)

        if save:  
            nib.save(bbox_img, save_path)
        return bbox_img

    def to_antsImage(self, data, save=False, save_path=None):

        img = ants.image_read(self.path)
        ants_img = ants.from_numpy(data, spacing=img.spacing, origin=img.origin, direction=img.direction)

        return ants_img

