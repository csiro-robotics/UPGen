"""
Helper functions and utilities
"""

from datetime import datetime as dt

from mrcnn import visualize
import numpy as np
import os

import cv2 

TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"

class Logger(object):
    """
    Log events and information to a file
    """
    def __init__(self, savePath):
        self.savePath = savePath
        self.log_file = open(self.savePath, 'a')
        self.log_line("Start of Log File")

    def close(self):
        self.log_line("End of Log File")
        self.log_file.close()

    def flush(self):
        self.log_file.flush()

    def time_stamp(self):
        now = dt.now()
        date_time = now.strftime(TIMESTAMP_FORMAT)
        self.log_file.write(date_time + ': ')

    def log_line(self, *args):
        '''
        Write each thing to the log file
        '''
        self.time_stamp()

        for log_item in args:
            self.log_file.write(str(log_item) + ' ')
        self.log_file.write('\n')

        self.flush()

    def log(self, *args):
        '''
        Write each thing to the log file
        '''
        self.time_stamp()

        for log_item in args:
            self.log_file.write(str(log_item) + ' ')
        
        self.flush()

    def newline(self):
        self.log_file.write("\n")
        self.flush()

def mask_to_rgb(mask):
    """
    Converts a mask to RGB Format
    """
    colours = visualize.random_colors(mask.shape[2])
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))

    for i in range(mask.shape[2]):
        for c in range(3):
            rgb_mask[:, :, c] = np.where(mask[:, :, i] != 0, int(colours[i][c] * 255), rgb_mask[:, :, c])

    return rgb_mask

def mask_to_outlined(mask):
    """
    Converts a mask to RGB Format
    """
    colours = visualize.random_colors(mask.shape[2])
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))

    for i in range(mask.shape[2]):
        for c in range(3):
            rgb_mask[:, :, c] = np.where(mask[:, :, i] != 0, int(colours[i][c] * 255), rgb_mask[:, :, c])

    # put edges over the top of the colours
    for i in range(mask.shape[2]):
        # Find the contour of the leaf
        threshold = mask[:, :, i]
        threshold[threshold != 0] = 255
        _, contours, hierarchy = cv2.findContours(threshold.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # Draw outline on mask
        if len(contours) > 0:
            cv2.drawContours(rgb_mask, [contours[0]], 0, (255, 255, 255), thickness=1)

    return rgb_mask

def check_create_dir(directory):
    if not os.path.isdir(directory):
        print("creating directory:", directory)
        os.mkdir(directory)
        return True
    return False