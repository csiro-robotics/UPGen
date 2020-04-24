"""
Provides implementations of different segmentation evaluation metrics
"""
import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

def flatten_mask(mask):
    """
    Converts a mask into a 1D array of incremental mask values
    """
    flat_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for i in range(mask.shape[2]):
        flat_mask[mask[:, :, i] != 0] = i + 1

    return flat_mask

def expand_mask(mask):
    """
    Converts mask from rgb to one slice per object.
    Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    """
    from skimage import color
    mask = color.rgb2grey(mask)

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
    smallMaskThresh = 10
    validMaskIndexes = []
    
    for i in range(expandedMask.shape[2]):
        if np.sum(expandedMask[:,:,i]) > smallMaskThresh:
            validMaskIndexes.append(i)
            
    filteredExpandedMask = np.zeros((mask.shape[0], mask.shape[1], len(validMaskIndexes)))
    
    for maskNum, maskIdx in enumerate(validMaskIndexes):
        filteredExpandedMask[:, :, maskNum] = expandedMask[:,:,maskIdx]

    return expandedMask

class BaseMetric(object):
    """
    The base class for metrics to follow
    """

    @staticmethod
    def calculate(prediction, ground_truth):
        """
        Return the calculated metric
        @return tuple (metric, std)
        """
        raise NotImplementedError

class SymmetricBestDice(BaseMetric):
    """
    Implementation of the Symmetric Best Dice
    """

    @staticmethod
    def dice_similarity(mask, gt_mask, seg_val=1):
        """Computes the DICE similarity between two masks.
        mask: predicted mask
        gt_mask: ground truth mask
        seg_val: mask value where backround is 0, default is 1

        Returns: Float of dice similarity between the two masks.

        Reference: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        """
        return np.sum(mask[gt_mask==seg_val])*2.0 / (np.sum(mask) + np.sum(gt_mask))

    @staticmethod
    def compute_dice(gt_masks, pred_masks):
        """Compute the average mask DICE similarity.

        Returns:
        mDICE: Mean DICE Similarity
        std_dice: std of above mean
        # TODO precisions: List of precisions at different class score thresholds.
        # TODO recalls: List of recall values at different class score thresholds.
        """
        
        DICEs = []
        
        # Visualise all masks
        """
        allMasks = np.zeros(gt_masks.shape[:2])
        for m_idx in range(gt_masks.shape[2]):
            plt.imshow(gt_masks[:, :, m_idx])
            plt.show()
            allMasks[gt_masks[:, :, m_idx] == 1] = (254 * (m_idx/gt_masks.shape[2]) + 1)

        plt.imshow(allMasks)
        plt.show()
        """    

        for pred_mask_idx in range(pred_masks.shape[2]):
            # Find index of gt_mask with highest DICE similary which is not taken
            best_gt = (0, None) # Tuple of diceSimilarity, gt_mask_index
                
            for gt_mask_idx in range(gt_masks.shape[2]):

                dice_sim = SymmetricBestDice.dice_similarity(pred_masks[:, :, pred_mask_idx], 
                                        gt_masks[:, :, gt_mask_idx])
                if dice_sim > best_gt[0]:
                    best_gt = (dice_sim, gt_mask_idx)
            """
            # visualise best match 
            try: # Catch any cases where there were more predictions than ground truths
                fig, axarr = plt.subplots(1,2)
                axarr[0].imshow(pred_masks[:, :, pred_mask_idx])
                axarr[1].imshow(gt_masks[:, :, best_gt[1]])
            except:
                fig, axarr = plt.subplots(1,2)
                axarr[0].imshow(pred_masks[:, :, pred_mask_idx])
                axarr[1].imshow(np.zeros(gt_masks.shape[:2]))
                print("no more ground truths, extra predictions")
                continue

            print("DICE", best_gt[0])
            plt.show()
            """

            # Handle no mask matched, too many objected detected
            # assigned DICE score of 0, the worst
            DICEs.append(best_gt[0])

        mDICE = np.mean(DICEs)
        std_dice = np.std(DICEs)

        return mDICE, std_dice

    @staticmethod
    def calculate(prediction, ground_truth):
        """
        Compute the symmetric average mask DICE similarity.

        To match the CVPPP evaluation metric.
        see: https://link.springer.com/article/10.1007/s00138-015-0737-3

        Returns:
        mDICE: symetric DICE Similarity
        std_dice: std of above mean
        """

        print("WARNING\nTHIS IMPLEMENTATION DOES NOT PRODUCE RESULTS THE SAME AS THE CVPPP COMPETITION")
        dice1 = SymmetricBestDice.compute_dice(ground_truth, prediction)
        dice2 = SymmetricBestDice.compute_dice(prediction, ground_truth)

        if dice2[0] < dice1[0]:
            return dice2
        else:
            return dice1

class CvpppSymmetricBestDice(BaseMetric):
    """
    Implementation of the Symmetric Best Dice by the CVPPP team
    @see https://competitions.codalab.org/competitions/18405#learn_the_details-evaluation
    """

    @staticmethod
    def DiffFGLabels(inLabel,gtLabel):
    # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
    #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
    # output: difference of the number of foreground labels

        # check if label images have same size
        if (inLabel.shape!=gtLabel.shape):
            return -1

        maxInLabel = np.int(np.max(inLabel)) # maximum label value in inLabel
        minInLabel = np.int(np.min(inLabel)) # minimum label value in inLabel
        maxGtLabel = np.int(np.max(gtLabel)) # maximum label value in gtLabel
        minGtLabel = np.int(np.min(gtLabel)) # minimum label value in gtLabel

        return  (maxInLabel-minInLabel) - (maxGtLabel-minGtLabel)

    @staticmethod
    def BestDice(inLabel,gtLabel):
    # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
    #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
    # output: score: Dice score
    #
    # We assume that the lowest label in inLabel is background, same for gtLabel
    # and do not use it. This is necessary to avoid that the trivial solution, 
    # i.e. finding only background, gives excellent results.
    #
    # For the original Dice score, labels corresponding to each other need to
    # be known in advance. Here we simply take the best matching label from 
    # gtLabel in each comparison. We do not make sure that a label from gtLabel
    # is used only once. Better measures may exist. Please enlighten me if I do
    # something stupid here...

        score = 0 # initialize output

        # check if label images have same size
        if (inLabel.shape!=gtLabel.shape):
            return score

        maxInLabel = np.max(inLabel) # maximum label value in inLabel
        minInLabel = np.min(inLabel) # minimum label value in inLabel
        maxGtLabel = np.max(gtLabel) # maximum label value in gtLabel
        minGtLabel = np.min(gtLabel) # minimum label value in gtLabel

        if(maxInLabel==minInLabel): # trivial solution
            return score

        for i in range(minInLabel+1,maxInLabel+1): # loop all labels of inLabel, but background
            sMax = 0; # maximum Dice value found for label i so far
            for j in range(minGtLabel+1,maxGtLabel+1): # loop all labels of gtLabel, but background
                s = CvpppSymmetricBestDice.Dice(inLabel, gtLabel, i, j) # compare labelled regions
                # keep max Dice value for label i
                if(sMax < s):
                    sMax = s
            score = score + sMax; # sum up best found values
        score = score/(maxInLabel-minInLabel)
        return score

    @staticmethod
    def FGBGDice(inLabel,gtLabel):
    # input: inLabel: label image to be evaluated. Background label is assumed to be the lowest one.
    #        gtLabel: ground truth label image. Background label is assumed to be the lowest one.
    # output: Dice score for foreground/background segmentation, only.

        # check if label images have same size
        if (inLabel.shape!=gtLabel.shape):
            return 0

        minInLabel = np.min(inLabel) # minimum label value in inLabel
        minGtLabel = np.min(gtLabel) # minimum label value in gtLabel

        one = np.ones(inLabel.shape)
        inFgLabel = (inLabel != minInLabel*one)*one
        gtFgLabel = (gtLabel != minGtLabel*one)*one

        return CvpppSymmetricBestDice.Dice(inFgLabel,gtFgLabel,1,1) # Dice score for the foreground

    @staticmethod
    def Dice(inLabel, gtLabel, i, j):
    # calculate Dice score for the given labels i and j

        # check if label images have same size
        if (inLabel.shape!=gtLabel.shape):
            return 0

        one = np.ones(inLabel.shape)
        inMask = (inLabel==i*one) # find region of label i in inLabel
        gtMask = (gtLabel==j*one) # find region of label j in gtLabel
        inSize = np.sum(inMask*one) # cardinality of set i in inLabel
        gtSize = np.sum(gtMask*one) # cardinality of set j in gtLabel
        overlap= np.sum(inMask*gtMask*one) # cardinality of overlap of the two regions
        if ((inSize + gtSize)>1e-8):
            out = 2*overlap/(inSize + gtSize) # Dice score
        else:
            out = 0

        return out

    @staticmethod
    def AbsDiffFGLabels(inLabel,gtLabel):
    # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
    #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
    # output: Absolute value of difference of the number of foreground labels

        return np.abs( CvpppSymmetricBestDice.DiffFGLabels(inLabel,gtLabel) )

    @staticmethod
    def calculate(prediction, ground_truth):
        '''
        Convert the masks into the form expected by the CVPPP eval code
        Then compute DICE
        '''
        gt_mask = flatten_mask(ground_truth)
        pred_mask = flatten_mask(prediction)

        # return std as 0 since it is not calculated
        return CvpppSymmetricBestDice.BestDice(pred_mask, gt_mask), 0

class DifferenceInCount(BaseMetric):
    """
    Implementation of difference in leaf count (DiC)
    """

    @staticmethod
    def calculate(prediction, ground_truth):
        '''
        Compute difference in count
        '''
        gt_mask = flatten_mask(ground_truth)
        pred_mask = flatten_mask(prediction)

        # Both will contain background unique values of 0 so can leave it in
        diff = len(np.unique(gt_mask)) - len(np.unique(pred_mask))

        # return std as it is not applicable
        return diff, 0

class AbsoluteDifferenceInCount(BaseMetric):
    """
    Implementation of abs difference in leaf count (|DiC|)
    """

    @staticmethod
    def calculate(prediction, ground_truth):
        """
        Calc abs of DiC
        """
        return abs(DifferenceInCount.calculate(prediction, ground_truth)[0]), 0

if __name__ == '__main__':
    # Test code
    from skimage import io

    im_path = '/media/war438/DATA/bg-cvppp_leaf-hf_cam-fixed_distractors-no_scale-14/train/plant00001_rgb.png'
    gt_path = '/media/war438/DATA/bg-cvppp_leaf-hf_cam-fixed_distractors-no_scale-14/train/plant00001_label.png'
    pred_path = '/home/war438/universal_leaf_segmenter/src/testOut3/plant00001_rgb.png'

    # Convert masks into format that mask rcnn would output
    gt = expand_mask(io.imread(gt_path))
    pred = expand_mask(io.imread(pred_path))

    print('CVPPP Dice', CvpppSymmetricBestDice.calculate(pred, gt))
    print('My Dice', SymmetricBestDice.calculate(pred, gt))
    print('DiC', DifferenceInCount.calculate(pred, gt))
    print('|Dic|', AbsoluteDifferenceInCount.calculate(pred, gt))