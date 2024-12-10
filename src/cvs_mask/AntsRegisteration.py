
import ants
import numpy as np

class ANTsRegistration:
    def __init__(self, fix_img_path, move_img_path):
        self.fix_img = ants.image_read(fix_img_path)
        self.move_img = ants.image_read(move_img_path)

    def image_registration(self, save=False, output_path=None):

        fixed_image = self.fix_img
        moving_image = self.move_img
        # Perform registration 
        registration_result = ants.registration(fixed=fixed_image, 
                                            moving=moving_image, 
                                            type_of_transform='Affine',
                                            grad_step=0.15,
                                            reg_iterations=(100, 70, 40),
                                            random_seed=35,
                                            )
        
        # apply transformation matrix to mask
        transformed_img = ants.apply_transforms(fixed=fixed_image, moving=moving_image,
                                                transformlist=registration_result['fwdtransforms'])
        if save:
            ants.image_write(transformed_img, output_path)

        return transformed_img

    def mask_registration(self, move_mask_path, save=False, output_path=None):
        """
        cut of skull area in ct image, mask image contain only brain area
        apply affine registration to no constract image and template image and do the transformation on mask image
        :param arterial_path: non contrast image path
        :param template_path: template image path
        :param mask_path: mask path
        :param mask_output: output path for brain area image
        :save: flag for saving intermidite image (clipped image and transformed non contrat image)
        :return: brain area image
        """

        fixed_image = self.fix_img
        moving_image = self.move_img
        mask_image = ants.image_read(move_mask_path)

        
        # clip value under 30 to 0 for ants intensity registration (value of ct image not from 0, but template image start from 0)
        #fixed_image_arr = fixed_image.numpy()
        #np.clip(fixed_image_arr, 30, np.max(fixed_image_arr), out=fixed_image_arr)
        #fixed_image_clip = ants.from_numpy(fixed_image_arr, spacing=fixed_image.spacing, origin=fixed_image.origin, direction=fixed_image.direction)
        
        # Perform registration
        registration_result = ants.registration(fixed=fixed_image, 
                                            moving=moving_image, 
                                            type_of_transform='Affine',
                                            grad_step=0.15,
                                            reg_iterations=(100, 70, 40),
                                            random_seed=35,
                                            )
        # apply transformation matrix to mask
        transformed_mask = ants.apply_transforms(fixed=fixed_image, moving=mask_image,
                                                transformlist=registration_result['fwdtransforms'])
        
        # save output
        if save:
            #ants.image_write(fixed_image_clip, 'clipped_arterial_image.nii.gz')
            #ants.image_write(registration_result['warpedmovout'], './registration_template.nii.gz')

            ants.image_write(transformed_mask, output_path)

        return transformed_mask



