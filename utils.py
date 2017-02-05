import os

import cv2
import numpy as np
from PIL import Image


def files_in_directory(directory, allowed_extensions):
    import os
    # print 'Reading all files in directory:', directory
    
    for file_or_dir in os.listdir(directory):
        full_path_to_file_or_dir = os.path.join(directory, file_or_dir)

        if not os.path.isfile(full_path_to_file_or_dir):
            continue
        
        file_extension = full_path_to_file_or_dir.lower().split('.')[-1]
        if file_extension in allowed_extensions:
            yield full_path_to_file_or_dir


def get_blob_bounding_boxes(input_image):
    """
    Input is grayscale image with white blobs
    Output is list of bounding boxes for these blobs
    """

    # Normalize image to values between 0 and 255
    image = input_image - input_image.min()
    # Not a shorthand operator because
    # https://github.com/numpy/numpy/issues/7225
    image = image * 255.0 / image.max()
    image = image.astype('uint8')

    # Given threshold value is ignored,
    # optimal threshold is computed using Otsu's method
    _, thresholded_image = cv2.threshold(
        image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Get outer contours of blobs as sets of points
    _, contours, _ = cv2.findContours(
        thresholded_image, mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding box for each contour
    bounding_boxes = [cv2.boundingRect(points) for points in contours]

    return bounding_boxes


# TODO: Add data augmentations
class DataReader:
    # 0 -> no fin, 1 -> fin
    CLASSES = {0: [0, 1], 1: [1, 0]}

    def __init__(self, data_dir, batch_size=1, file_names=False):
        self.data_arrays = []
        self.current = 0
        self.batch_size = batch_size
        self.image_names = file_names

        for item in os.listdir(data_dir):
            image = Image.open(data_dir + item)
            image_array = np.asarray(image)

            if len(image_array.shape) == 2:
                image_array = np.expand_dims(image_array, axis=2)

            # TODO: Generic class determination logic
            if "no_fin" in item:
                class_id = 0
            else:
                class_id = 1

            if self.image_names:
                self.data_arrays.append((image_array,
                                         self.CLASSES[class_id], item))
            else:
                self.data_arrays.append((image_array, self.CLASSES[class_id]))

        self.data_count = len(self.data_arrays)

        np.random.shuffle(self.data_arrays)

    def next(self):
        batch = []
        images_batch = []
        labels_batch = []
        image_names_batch = []

        for _ in range(self.batch_size):

            images_batch.append(self.data_arrays[self.current][0])
            labels_batch.append(self.data_arrays[self.current][1])

            batch = [images_batch, labels_batch]

            if self.image_names:
                image_names_batch.append(self.data_arrays[self.current][2])

                batch = batch + [image_names_batch]

            if self.current == len(self.data_arrays) - 1:
                self.current = 0
            else:
                self.current += 1

        return batch



