"""
The dataset API for the CVPPP dataset

Expected dataset format:
. Should contain 2 folders "train" and "crossVal". \
                        Each should contain images "plant<num>_rgb.png" and "plant<num>_label.png"

DataDir
    train
        plant<num>_rgb_.png
        plant<num>_label.png
        plant000_rgb_.png
        plant000_label.png
        ...
    crossVal
        plant<num>_rgb_.png
        plant<num>_label.png
        plant002_rgb_.png
        plant002_label.png
        ...
"""

import sys
import os
import matplotlib
if sys.platform == 'linux' and 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
import cv2

from mrcnn import utils


CLASS_NAME = 'cvppp'
OBJECT_CLASS_NAME = 'leaf'


class CVPPP_Dataset(utils.Dataset):
    """
    Loads the CVPP dataset to play with
    """
    def __init__(self, blur_images=False, class_map=None, image_type='png'):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        self.blur_images = blur_images

        self.image_type = image_type

    def load_cvppp(self, dataset_dir, subset):
        """
        Loads the images from the subset
        """
        import glob
        image_dir = os.path.join(dataset_dir, subset)

        if not os.path.isdir(image_dir):
            raise OSError('Invalid Data Directory at: ' + image_dir)

        self.add_class(CLASS_NAME, 1, OBJECT_CLASS_NAME)

        image_id = 0
        for i, fname in enumerate(glob.glob(os.path.join(image_dir, '*_rgb.' + self.image_type))):
            
            # Check if mask is empty, if so skip it
            if not self.check_mask(fname.replace('rgb', 'label')):
                continue
                
            self.add_image(CLASS_NAME, image_id, fname,
                          mask_path=fname.replace('rgb', 'label'))
            image_id += 1

        if image_id == 0:
            raise OSError('No training images in: ' + image_dir)
        
    def check_mask(self, mask_path):
        try:
            mask = io.imread(mask_path)
            if mask.sum() == 0:
                print("Skipping calculated empty mask:", mask_path)
                return False
        except:
            print("Skipping mask error:", mask_path)
            return False
        
        return True

    def load_mask(self, image_id):
        """
        Loads an image mask from its id
        
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a CVPP image, delegate to parent class.
        image_info = self.image_info[image_id]
        mask = color.rgb2grey(io.imread(image_info['mask_path']))

        # Remove alpha channel if it exists
        if mask.ndim == 4:
            mask = mask[:, :, :3]

        leaves = np.unique(mask)
        # -1 to skip background, 0
        expandedMask = np.zeros((mask.shape[0], mask.shape[1], len(leaves)-1))
        for i in range(1, len(leaves)):
            l = mask.copy()
            l[l!=leaves[i]] = False
            l[l!=0] = True
            expandedMask[:,:,i-1] = l

        # Filter out masks smaller than a smallMaskThresh
        smallMaskThresh = 3
        validMaskIndexes = []
        
        for i in range(expandedMask.shape[2]):
            if np.sum(expandedMask[:,:,i]) > smallMaskThresh:
                validMaskIndexes.append(i)
                
        filteredExpandedMask = np.zeros((mask.shape[0], mask.shape[1], len(validMaskIndexes)))
        
        for maskNum, maskIdx in enumerate(validMaskIndexes):
            filteredExpandedMask[:, :, maskNum] = expandedMask[:,:,maskIdx]

        class_ids = np.array([1] * expandedMask.shape[2])
        return expandedMask, class_ids.astype(np.int32)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
            If it is an RGBA image, drop the alpha channel
        """
        # Load image
        image = io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = color.gray2rgb(image)

        if image.shape[2] == 4:
            image = image[:, :, :3]
        
        if self.blur_images:
            image = self.gaussian_blur(image.astype(np.uint8))
        return image.astype(np.uint8)
    
    def gaussian_blur(self, image):
        '''
        apply a gaussian blur to the image.
            5x5 kernel, sigmaX 3, sigmaY (0 means set equal to sigmaX) 0
        '''
        return cv2.GaussianBlur(image, (5, 5), 3, 0)

    def visualise_mask(self, image_id):
        # If not a CVPP image
        image_info = self.image_info[image_id]
        if image_info["source"] != CLASS_NAME:
            return

        mask = color.rgb2grey(io.imread(image_info['mask_path']))
        print(type(mask))
        leaves = np.unique(mask)
        # -1 to skip background, 0
        expandedMask = np.zeros((mask.shape[0], mask.shape[1], len(leaves)-1))
        print(expandedMask.shape, mask.shape, leaves)
        for i in range(1, len(leaves)):
            l = mask.copy()
            l[l!=leaves[i]] = False
            l[l!=0] = True
            expandedMask[:,:,i-1] = l
            plt.figure()
            plt.imshow(expandedMask[:,:,i-1])

        plt.figure()
        plt.imshow(mask)
        plt.show()

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == CLASS_NAME:
            return info
        else:
            return info # Naugthy
            super(self.__class__).image_reference(self, image_id)

class Fine_Tune_CVPPP_Dataset(CVPPP_Dataset):
    """
    Loads the CVPP dataset to play with but only loads a number of images 
    """
    def __init__(self, blur_images=False, class_map=None, image_type='png', num_imgs=-1):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        self.blur_images = blur_images

        self.image_type = image_type

        self.num_imgs = num_imgs

    def load_cvppp(self, dataset_dir, subset):
        """
        Loads the images from the subset
        """
        import glob
        image_dir = os.path.join(dataset_dir, subset)

        if not os.path.isdir(image_dir):
            raise OSError('Invalid Data Directory at: ' + image_dir)

        self.add_class(CLASS_NAME, 1, OBJECT_CLASS_NAME)

        image_id = 0
        image_files = glob.glob(os.path.join(image_dir, '*_rgb.' + self.image_type))

        if self.num_imgs > 0:
            num_images = min(self.num_imgs, len(image_files) - 1)

            np.random.shuffle(image_files) # Inplace shuffle
            image_files = image_files[:num_images]

        for i, fname in enumerate(image_files):
            # Check if mask is empty, if so skip it
            if not self.check_mask(fname.replace('rgb', 'label')):
                continue

            self.add_image(CLASS_NAME, image_id, fname,
                          mask_path=fname.replace('rgb', 'label'))
            image_id += 1

        if image_id == 0:
            raise OSError('No training images in: ' + image_dir)

if __name__ == '__main__':
    dataset_path = '/media/war438/DATA/bg-cvppp_leaf-hf_cam-fixed_distractors-no_scale-14'
    print("Attempting to visualise some data at", dataset_path, "\n")

    dataset = CVPPP_Dataset()
    dataset.load_cvppp(dataset_path, 'crossVal')

    print("Visualising image:", str(dataset.image_reference(0)), "\n")
    dataset.visualise_mask(0)
    
    sample_image = dataset.load_image(0)
    plt.imshow(sample_image)
    plt.show()
