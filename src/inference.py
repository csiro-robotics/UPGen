"""
Predict segmentations. Input a model and images to obtain segmentations for the images
"""
import os
import sys
from glob import glob
import argparse
import time

import config_cvppp
import dataset_cvppp
from utils import Logger, mask_to_rgb, mask_to_outlined

from mrcnn import model, visualize

from imgaug import augmenters as iaa
from skimage import io
import numpy as np
import cv2

import matplotlib
if sys.platform == 'linux' and 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
IMAGENET_WEIGHTS_PATH = ''
COCO_WEIGHTS_PATH = '/home/war438/universal_leaf_segmenter/library/Mask_RCNN_weights/mask_rcnn_coco.h5'

CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'

def arguments():
    """
    CMD args
    """
    parser = argparse.ArgumentParser(description='Performs inference using a Mask RCNN Model')
    parser.add_argument('--dataPattern', type=str, required=True,
                        help="A glob file path pattern in quotations. e.g. '/data/plants/plant????_*_image.png'")
    parser.add_argument('--outputDir', type=str, required=True,
                        help='Directory to save all outputs to')
    parser.add_argument('--weightsPath', type=str, required=True,
                        help='Path to model weights to use (h5 file)')
    parser.add_argument('--blurImages', dest='blurImages', action='store_true')
    parser.add_argument('--dontBlurImages', dest='blurImages', action='store_false')
    parser.set_defaults(blurImages=False)

    parser.add_argument('--verboseDetection', dest='verbose', action='store_true')
    parser.add_argument('--quietDetection', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

    parser.add_argument('--useCPU', dest='useCPU', action='store_true')
    parser.add_argument('--useGPU', dest='useCPU', action='store_false')
    parser.set_defaults(useCPU=False)

    parser.add_argument('--blendOccluded', dest='alphaBlend', action='store_true')
    parser.add_argument('--overwriteOccluded', dest='alphaBlend', action='store_false')
    parser.set_defaults(alphaBlend=False)

    parser.add_argument('--imgPattern', type=str, default='',
                        help="Option to supply the filename pattern and just a path for the dataPattern arg")

    return parser.parse_args()

def blur(image):
    """
    For blurring images
    """
    # TODO make these values params
    kv = 101
    return cv2.GaussianBlur(image, (kv, kv), 92, 0)

def load_image(im_path):
    image = io.imread(im_path)
    # Check for alpha channel
    if image.shape[2] > 3:
        image = image[:, :, :3]

    return image

def predict_segmentations():
    """
    Main function to perform segmentation prediction
    """
    args = arguments()

    # Assemble image pattern if required
    if args.imgPattern != '':
        image_pattern = os.path.join(args.dataPattern, args.imgPattern)
    else:
        image_pattern = args.dataPattern

    print("Image Pattern:", image_pattern)

    # Create output dir
    assert not os.path.isdir(args.outputDir), "output dir already exists"
    os.mkdir(args.outputDir)

    # Init config
    configuration = config_cvppp.InferenceConfig()

    if args.useCPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Init model
    inference_model = model.MaskRCNN(mode="inference", 
                          config=configuration,
                          model_dir=args.outputDir)

    assert os.path.exists(args.weightsPath), "Weights file does not exist at " + args.weightsPath
    inference_model.load_weights(args.weightsPath, by_name=True)

    # Predict Images
    with open(os.path.join(args.outputDir, 'leafCounts.csv'), 'a') as count_file:
        count_file.write("Image, Count\n")
        for im_path in glob(image_pattern):
            out_path = os.path.join(args.outputDir, os.path.basename(im_path))

            print("Saving prediction for", im_path, "at", out_path)

            try:
                image = load_image(im_path)
            except:
                print("Bad File for prediction:", im_path)
                continue

            if args.blurImages:
                image = blur(image)

            results = inference_model.detect([image], verbose=args.verbose)

            if args.alphaBlend:
                rgb_mask = mask_to_outlined(results[0]['masks'])
            else:
                rgb_mask = mask_to_rgb(results[0]['masks'])
            
            io.imsave(out_path, rgb_mask.astype(np.uint8))



            count_file.write(os.path.basename(im_path) + ", " + str(results[0]['masks'].shape[2]) + "\n")

if __name__ == '__main__':
    predict_segmentations()
