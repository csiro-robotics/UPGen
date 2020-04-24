"""
The Mask RCNN config file for the CVPPP dataset
"""

from mrcnn import config

class TrainConfig(config.Config):
    NAME = 'cvppp'
    NUM_CLASSES = 1 + 1 # background + leaf
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    # og: 0.3, reduced: 0.2
    DETECTION_NMS_THRESHOLD = 0.3

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 256

    # Use a small epoch since the data is simple
    # From keras documentation: ceil(num_samples / batch_size) https://keras.io/models/sequential/#fit_generator
    # Since we want to report more regularly than a full epoch for dataset of 20k images, set this to 1000 (effectively 1 epoch is 4000 images)
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 500

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 4 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size (DEFAULT)
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False

    def to_string(self):
        """Compile String Of Configuration values."""
        s = "Configurations:"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                s = s + "\n" + "{:30} {}".format(a, getattr(self, a))
        s = s + "\n"
        return s

class InferenceConfig(TrainConfig): 
    IMAGES_PER_GPU = 1

if __name__ == '__main__':
    cfg = TrainConfig()
    cfg.display()
