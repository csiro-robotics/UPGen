"""
A script which will train a Mask RCNN model with the data provided
"""

import os
import sys
from glob import glob
import argparse
import time

import config_cvppp
import dataset_cvppp
from utils import Logger, mask_to_rgb

from mrcnn import model, visualize

from imgaug import augmenters as iaa
from skimage import io
import numpy as np

import matplotlib
if sys.platform == 'linux' and 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Directory names
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'

def arguments():
    """
    CMD args
    """
    parser = argparse.ArgumentParser(description='Trains a Mask RCNN Model')
    parser.add_argument('--dataDir', type=str, required=True,
                        help='Directory containing training data stored in the expected format. See dataset_cvppp.py')
    parser.add_argument('--matchedTraing2ndDataDir', type=str, default='',
                        help='2nd training Directory containing training data stored in the expected format. See dataset_cvppp.py')
    parser.add_argument('--outputDir', type=str, required=True,
                        help='Directory to save all outputs to')
    parser.add_argument('--name', type=str, default='MaskRCNN_exp',
                        help='Experiment name')
    parser.add_argument('--description', type=str, default='Default experiment description',
                        help='A note/description of the training')
    parser.add_argument('--init', type=str, required=True,
                        help='The initial weights of the network. Can be a path, "rand", or "last" for specific weights, random \
                              initialisation, or the last checkpoint if restarting an experiment')

    parser.add_argument('--numSamples', type=int, default=-1,
                        help='Number of samples randomly selected and used from each training dataset. -1 implies use everything')

    parser.add_argument('--numEpochs', type=int, required=True,
                        help='Number of training epochs')
    parser.add_argument('--blurImages', dest='blurImages', action='store_true')
    parser.add_argument('--dontBlurImages', dest='blurImages', action='store_false')
    parser.set_defaults(blurImages=False)

    parser.add_argument('--evaluateDataCallback', dest='evaluateDataCallback', action='store_true')
    parser.set_defaults(evaluateDataCallback=False)

    parser.add_argument('--savePredictionCallback', dest='savePredictionCallback', action='store_true')
    parser.set_defaults(savePredictionCallback=False)

    return parser.parse_args()

def create_output_dir(output_dir):
    """
    Creates the various output directories
    """
    if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

    for folder in [CHECKPOINT_DIR, LOG_DIR]:
        folder_path = os.path.join(output_dir, folder)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

def get_augmentation_sequence():
    """
    Define the augmentation for training
    """
    # Macro to apply something with 50% chance
    sometimes = lambda aug: iaa.Sometimes(0.5, aug) # 50%
    rarely = lambda aug: iaa.Sometimes(0.1, aug) # 10%

    # Augmentation applied to every image
    # Augmentors sampled one value per channel
    aug_sequence = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 50% of all images

            # crop images by -0.25% to 0.25% of their height/width
            # positive values crop the image, negative pad
            sometimes(iaa.CropAndPad(
                percent=(-0.25, 0.25),
                pad_mode=['constant', 'edge'], # pad with constant value of the edge value
                pad_cval=(0, 0)  # if mode is constant, use a cval between 0 and 0 to ensure mask background is preserved
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 0), # if mode is constant, use a cval between 0 and 0 to ensure mask background is preserved
                mode='constant' # ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # rarely(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
            iaa.GaussianBlur((0, 3.0)),
            iaa.Add((-10, 10), per_channel=0.7), # change brightness of images (by -10 to 10 of original value)
            iaa.AddToHueAndSaturation((-20, 20)),
            # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        ],
        random_order=True
    )

    return aug_sequence

def run_example_augmentations():
    """
    Function which can be called to visualise some example augmentations and ensure that they are valid
    This uses the image loading function inside maskRCNN so the user can be confident in the data being fed to the network
    """
    parser = argparse.ArgumentParser(description='Visualise example augmentations')
    parser.add_argument('--dataDir', type=str, required=True,
                        help='Directory containing training data stored in the expected format. See dataset_cvppp.py')
    parser.add_argument('--outputDir', type=str, required=True,
                        help='Directory to save example images to')
    parser.add_argument('--numImages', type=int, default=30,
                        help='How many images to save')
    parser.add_argument('--blurImages', dest='blurImages', action='store_true')
    parser.add_argument('--dontBlurImages', dest='blurImages', action='store_false')
    parser.set_defaults(blurImages=False)

    args = parser.parse_args()

    # Create output dir
    assert not os.path.isdir(args.outputDir), "output dir already exists"
    os.mkdir(args.outputDir)

    # # Init dataset
    train_dataset = dataset_cvppp.CVPPP_Dataset()
    
    train_dataset.load_cvppp(args.dataDir, 'train')
    train_dataset.prepare()

    # Init config
    configuration = config_cvppp.TrainConfig()

    # Init augmentation
    augmentation = get_augmentation_sequence()

    # Generate images
    for i in range(args.numImages):
        image, meta, class_ids, bbox, mask = model.load_image_gt(train_dataset, configuration, i, augmentation=augmentation)

        rgb_mask = mask_to_rgb(mask)

        im_path = os.path.join(args.outputDir, str(i) + '_image.png')
        mask_path = os.path.join(args.outputDir, str(i) + '_mask.png')
        io.imsave(im_path, image)
        io.imsave(mask_path, rgb_mask)

        print("Saved example", i)
    
def load_datasets(args):
    if args.matchedTraing2ndDataDir == '': # Load the 1 dataset
        # Load dataset API (Alread logged in the args log step)
        train_dataset = dataset_cvppp.Fine_Tune_CVPPP_Dataset(blur_images=args.blurImages, num_imgs=args.numSamples)
        train_dataset.load_cvppp(args.dataDir, 'train')
        train_dataset.prepare()

        crossVal_dataset = dataset_cvppp.Fine_Tune_CVPPP_Dataset(blur_images=args.blurImages) # Use all for crossVal
        crossVal_dataset.load_cvppp(args.dataDir, 'crossVal')
        crossVal_dataset.prepare()
    else: # Load both
        print("Using matched Training")
        train_dataset1 = dataset_cvppp.CVPPP_Dataset(blur_images=args.blurImages)
        train_dataset1.load_cvppp(args.dataDir, 'train')
        train_dataset1.prepare()

        crossVal_dataset1 = dataset_cvppp.CVPPP_Dataset(blur_images=args.blurImages)
        crossVal_dataset1.load_cvppp(args.dataDir, 'crossVal')
        crossVal_dataset1.prepare()

        train_dataset2 = dataset_cvppp.CVPPP_Dataset(blur_images=args.blurImages)
        train_dataset2.load_cvppp(args.matchedTraing2ndDataDir, 'train')
        train_dataset2.prepare()

        crossVal_dataset2 = dataset_cvppp.CVPPP_Dataset(blur_images=args.blurImages)
        crossVal_dataset2.load_cvppp(args.matchedTraing2ndDataDir, 'crossVal')
        crossVal_dataset2.prepare()

        train_dataset = (train_dataset1, train_dataset2)
        crossVal_dataset = (crossVal_dataset1, crossVal_dataset2)
    
    return train_dataset, crossVal_dataset

def train():
    """
    The main training procedure
    """
    args = arguments()

    # Create output directories
    create_output_dir(args.outputDir)

    # Start Log File
    log_path = os.path.join(args.outputDir, LOG_DIR, time.strftime('%Y-%m-%d_%H-%M-%S.log'))
    log_file = Logger(log_path)

    # Log arguments
    arg_str = ''
    for arg in vars(args):
        arg_str += "\n" + "{:30} {}".format(str(arg), getattr(args, arg))
    log_file.log_line("Arguments", arg_str)
    log_file.newline()

    # Load Params
    configuration = config_cvppp.TrainConfig()

    # Log params
    log_file.log_line("Config Parameters\n", configuration.to_string())
    log_file.newline()

    ## Load dataset API (Already logged in the args log step)
    train_dataset, crossVal_dataset = load_datasets(args)

    # Init the model
    checkpoint_path = os.path.join(args.outputDir, CHECKPOINT_DIR)
    training_model = model.MaskRCNN('training', configuration, checkpoint_path)

    # Load weights
    if args.init == 'last':
        weights_path = training_model.find_last()
        log_file.log_line("Initialised with ", weights_path)
        training_model.load_weights(weights_path, by_name=True)

    elif args.init == 'rand':
        log_file.log_line("Initialised with ", "random weights")
        pass

    else:
        if not os.path.exists(args.init):
            raise OSError('No weights at: ' + args.init)
        
        log_file.log_line("Initialised with ", args.init)
        training_model.load_weights(args.init, by_name=True)

    # Train the model
    augmentation = get_augmentation_sequence()

    custom_callbacks = None

    training_model.train(train_dataset, crossVal_dataset, 
            learning_rate=configuration.LEARNING_RATE, 
            epochs=args.numEpochs,
            augmentation=augmentation,
            layers='all',
            custom_callbacks=custom_callbacks) # Train all layers

    # Close the log file
    log_file.close()

if __name__ == "__main__":
    train()
    # run_example_augmentations()

