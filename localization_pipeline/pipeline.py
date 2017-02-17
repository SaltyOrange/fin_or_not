import sys
import os
import glob

import numpy
import cv2
from PIL import Image

#from localization_net1.predict import Predictor as LocalizationPredictor
from inception2layer_net.predict import Predictor as LocalizationPredictor
from classification_net.predict import Predictor as ClassificationPredictor
from utils import get_blob_bounding_boxes

class Pipeline:
    
    def __init__(self, localization_dir, classification_dir):
        self.BOUNDING_BOX_EXPAND_PERC = 0.2

        print 'Loading localization predictor'
        self.localization_predictor = LocalizationPredictor(localization_dir)
        print 'Loading classification predictor'
        self.classification_predictor = ClassificationPredictor(classification_dir)

    def predict(self, image_path):
        """ Return value is tuple 
            first value is bool -> Contains fin?
            second value is array with rectangles or empty array 
        """
        prediction, image_output = self.localization_predictor.predict(image_path, 1)
        contains_fin = prediction == [0]        

        classmap = numpy.array(image_output[0].convert('L'))

        if not contains_fin:
            return (False, [], classmap)
                
        proposed_rects = get_blob_bounding_boxes(classmap)

        # Transform rects to original image size
        image = cv2.imread(image_path)
        ch, cw = classmap.shape[:2]
        oh, ow = image.shape[:2]
        ratio_h, ratio_w = float(ch)/oh, float(cw) / ow

        proposed_rects = [(
            int(r[0]/ratio_w),
            int(r[1]/ratio_h),
            int(r[2]/ratio_w),
            int(r[3]/ratio_h)
        ) for r in proposed_rects]

        def expand_rect(rect):
            x, y, w, h = rect
            # Bounding box expansion
            x = max(0, int(x - x*self.BOUNDING_BOX_EXPAND_PERC/2))
            y = max(0, int(y - y*self.BOUNDING_BOX_EXPAND_PERC/2))
            w = int(w + w*self.BOUNDING_BOX_EXPAND_PERC)
            h = int(h + h*self.BOUNDING_BOX_EXPAND_PERC)

            if x + w > ow: w -= x+w-ow
            if y + h > oh: h -= y+w-oh

            return (x, y, w, h)
            
        proposed_rects = [expand_rect(rect) for rect in proposed_rects]

        # Now for each rect run classification
        def rect_contains_fin(rect):
            x, y, w, h = rect
            image_region = Image.fromarray(image[x:x+w, y:y+h])
            contains_fin = self.classification_predictor.predict(image_region) == [0]
            return contains_fin

        #rects_with_fins = filter(rect_contains_fin, proposed_rects)
        rects_with_fins = proposed_rects
        return (True, rects_with_fins, classmap)
