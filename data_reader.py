import os
import numpy as np

from PIL import Image

# TODO: Add data augmentations

class DataReader:
    # 0 -> no fin, 1 -> fin
    CLASSES = {0: [0, 1], 1: [1, 0]}

    def __init__(self, data_dir, batch_size=1, image_names=False):
        self.data_arrays = []
        self.current = 0
        self.batch_size = batch_size
        self.image_names = image_names

        for item in os.listdir(data_dir):
            image = Image.open(data_dir + item)
            image_array = np.asarray(image)

            # image_array = np.expand_dims(image_array, axis=0)
            image_array = np.expand_dims(image_array, axis=3)
            if "no_fin" in item:
                class_id = 0
            else:
                class_id = 1

            if self.image_names:
                self.data_arrays.append((image_array,
                                         self.CLASSES[class_id], item))
            else:
                self.data_arrays.append((image_array, self.CLASSES[class_id]))

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


