"""
Evaluates a model. Input a model, data and labels to obtain predictions and a performance score
"""
import os
import sys
from glob import glob
import argparse
import time

import matplotlib
if sys.platform == 'linux' and 'DISPLAY' not in os.environ:
    print("Using matplotlib backend: 'Agg' to plot on the HPC without an X server")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config_cvppp
import dataset_cvppp
from utils import Logger, mask_to_rgb
from metrics import DifferenceInCount, AbsoluteDifferenceInCount, CvpppSymmetricBestDice

from mrcnn import model, visualize, utils

from imgaug import augmenters as iaa
from skimage import io
import numpy as np
import cv2

# Names of things
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'

OVERALL_RESULTS_FILENAME = 'overallResults.csv'
RESULTS_FILENAME = 'sampleResults.csv'

def arguments():
    """
    CMD args
    """
    parser = argparse.ArgumentParser(description='Evaluates a Mask RCNN model on the provided data with ground truth labels')
    parser.add_argument('--dataDir', type=str, required=True,
                        help='Directory containing training data stored in the expected format. See dataset_cvppp.py')
    parser.add_argument('--outputDir', type=str, required=True,
                        help='Directory to save all outputs to')
    parser.add_argument('--weightsPath', type=str, required=True,
                        help='Path to model weights to use (h5 file)')

    parser.add_argument('--savePredictions', dest='savePredictions', action='store_true')
    parser.add_argument('--dontSavePredictions', dest='savePredictions', action='store_false')
    parser.set_defaults(savePredictions=False)

    parser.add_argument('--blurImages', dest='blurImages', action='store_true')
    parser.add_argument('--dontBlurImages', dest='blurImages', action='store_false')
    parser.set_defaults(blurImages=False)

    parser.add_argument('--verboseDetection', dest='verbose', action='store_true')
    parser.add_argument('--quietDetection', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

    return parser.parse_args()

class MetricTracker(object):
    def __init__(self):
        self.metrics = []
        self.stds = []
    
    def add(self, metric, std):
        self.metrics.append(metric)
        self.stds.append(std)

    def calc_mean(self):
        return np.mean(self.metrics), np.std(self.metrics)

def visualise_prediction(image, prediction, ground_truth, save_path, alpha=0.5):
    """
    Saves a visualisation of the prediction
    """
    prediction = mask_to_rgb(prediction)
    ground_truth = mask_to_rgb(ground_truth)

    # Plot Images and save
    fig, arr = plt.subplots(2, 3)
    # RGB image
    arr[0, 0].set_title('RGB')
    arr[0, 0].imshow(image)
    arr[0, 0].set_xticklabels([])
    arr[0, 0].set_yticklabels([])
    # Ground Truth
    arr[0, 1].set_title('Ground Truth')
    arr[0, 1].imshow(ground_truth)
    arr[0, 1].set_xticklabels([])
    arr[0, 1].set_yticklabels([])
    # Prediction
    arr[0, 2].set_title('Prediction')
    arr[0, 2].imshow(prediction)
    arr[0, 2].set_xticklabels([])
    arr[0, 2].set_yticklabels([])
    # Ground Truth
    arr[1, 1].set_title('Ground Truth')
    arr[1, 1].imshow(image)
    arr[1, 1].imshow(ground_truth, alpha=alpha)
    arr[1, 1].set_xticklabels([])
    arr[1, 1].set_yticklabels([])
    # Prediction
    arr[1, 2].set_title('Prediction')
    arr[1, 2].imshow(image)
    arr[1, 2].imshow(prediction, alpha=alpha)
    arr[1, 2].set_xticklabels([])
    arr[1, 2].set_yticklabels([])
    # Stats
    arr[1, 0].set_title('Stats')
    arr[1, 0].axis('off')
    # arr[1, 0].text(0.1, 0.3, "image")
    arr[1, 0].set_xticklabels([])
    arr[1, 0].set_yticklabels([])

    plt.savefig(save_path)
    plt.cla()
    plt.close(fig)

def evaluate_model():
    """
    The main evaluation procedure
    """
    args = arguments()

    # Create output dir
    assert not os.path.isdir(args.outputDir), "output dir already exists"
    os.mkdir(args.outputDir)

    # Init config
    configuration = config_cvppp.InferenceConfig()

    # Init model
    inference_model = model.MaskRCNN(mode="inference", 
                          config=configuration,
                          model_dir=args.outputDir)

    assert os.path.exists(args.weightsPath), "Weights file does not exist at " + args.weightsPath
    inference_model.load_weights(args.weightsPath, by_name=True)

    # Load dataset API
    test_dataset = dataset_cvppp.CVPPP_Dataset()

    if os.path.isdir(os.path.join(args.dataDir, 'test')):
        test_dataset.load_cvppp(args.dataDir, 'test')
    else:
        test_dataset.load_cvppp(args.dataDir, '') # Assume it is just this directory
    test_dataset.prepare()

    # init metrics
    dice = MetricTracker()
    DiC = MetricTracker()
    abs_DiC = MetricTracker()
    mAP = MetricTracker()

    # save predictions
    with open(os.path.join(args.outputDir, RESULTS_FILENAME), 'a') as results_file:
        results_file.write("Filename, Path, Dice, Dice (std), DiC, DiC (std), absDiC, absDiC (std), mAP, mAP (std)\n")

        for image_id in test_dataset.image_ids:
            # Activate the choice of image
            test_dataset.image_reference(image_id)

            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                model.load_image_gt(test_dataset, configuration, image_id, use_mini_mask=False)

            image_path = test_dataset.image_reference(image_id)['path']
            
            # Run inference
            results = inference_model.detect([image], verbose=args.verbose)
            r = results[0]

            results_file.write(os.path.basename(image_path) + ', ' + image_path + ', ')

            # Dice metric
            dice_res, dice_std = CvpppSymmetricBestDice.calculate(r['masks'], gt_mask)
            dice.add(dice_res, dice_std)
            results_file.write(str(dice_res) + ', ' + str(dice_std) + ', ')

            # DiC metric
            dic_res, dic_std = DifferenceInCount.calculate(r['masks'], gt_mask)
            DiC.add(dic_res, dic_std)
            results_file.write(str(dic_res) + ', ' + str(dic_std) + ', ')

            # abs DiC metric
            abs_dic_res, abs_dic_std = AbsoluteDifferenceInCount.calculate(r['masks'], gt_mask)
            abs_DiC.add(abs_dic_res, abs_dic_std)
            results_file.write(str(abs_dic_res) + ', ' + str(abs_dic_std) + ', ')

            # mAP @ IoU 0.5 metric
            mAPResult, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                        r["rois"], r["class_ids"], r["scores"], r['masks'])
            mAP.add(mAPResult, 0)
            results_file.write(str(mAPResult) + ', ' + str(0) + '\n')

            if args.savePredictions:
                # Save visualisation of prediction
                save_path = os.path.join(args.outputDir, os.path.basename(image_path))
                visualise_prediction(image, r['masks'], gt_mask, save_path)

    # save overall results
    with open(os.path.join(args.outputDir, OVERALL_RESULTS_FILENAME), 'a') as results_file:
        results_file.write("Metric, Result, STD\n")
        overall_dice = dice.calc_mean()
        results_file.write("Dice, " + str(overall_dice[0]) + ', ' + str(overall_dice[1]) + '\n')

        overall_dic = DiC.calc_mean()
        results_file.write("DiC, " + str(overall_dic[0]) + ', ' + str(overall_dic[1]) + '\n')

        overall_abs_dic = abs_DiC.calc_mean()
        results_file.write("|DiC|, " + str(overall_abs_dic[0]) + ', ' + str(overall_abs_dic[1]) + '\n')

        overall_mAP = mAP.calc_mean()
        results_file.write("mAP, " + str(overall_mAP[0]) + ', ' + str(overall_mAP[1]) + '\n')


if __name__ == '__main__':
    evaluate_model()
