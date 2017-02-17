import sys
import os
import glob

import numpy
import cv2
from PIL import Image

from inception2layer_net.predict import Predictor
from utils import get_blob_bounding_boxes

VALID_PREDICTIONS = [[0]]
BOUNDING_BOX_EXPAND_PERC = 0.2

if len(sys.argv) < 4:
    sys.exit("Usage: script.py localization_model_dir images_dir output_dir")

# Get localization predictor
predictor = Predictor(sys.argv[1])

# Get image dir
image_dir = sys.argv[2]

# Get output dir
output_dir = sys.argv[3]

for i, image_path in enumerate(glob.glob(os.path.join(image_dir, '*.jpg'))):
    # Get prediction and classmap for image
    prediction, images = \
        predictor.predict(image_path, return_type=1)
    
    if prediction in VALID_PREDICTIONS:
        classmap = images[0].convert("L")
        classmap = numpy.array(classmap)

        # Get bounding boxes for blobs in classmap
        bounding_boxes = get_blob_bounding_boxes(classmap)

        # Translate bounding boxes into original picture coordinate system
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        classmap_height, classmap_width = classmap.shape
        original_image_height, original_image_width, _ = original_image.shape
        width_ratio = original_image_width/classmap_width
        height_ratio = original_image_height/classmap_height	

        for j, box in enumerate(bounding_boxes):
            x, y, w, h = box
            # Bounding box coordinate translation
            x *= width_ratio
            w *= width_ratio
            y *= height_ratio
            h *= height_ratio   

            # Bounding box expansion TODO: Fix this
            x = max(0, int(x - x*BOUNDING_BOX_EXPAND_PERC/2))
            y = max(0, int(y - y*BOUNDING_BOX_EXPAND_PERC/2))
            w = int(w + w*BOUNDING_BOX_EXPAND_PERC)
            h = int(h + h*BOUNDING_BOX_EXPAND_PERC)
            if x + w > original_image_width: w -= x+w-original_image_width
            if y + h > original_image_height: h -= y+w-original_image_height

            image = Image.fromarray(original_image[y:y+h, x:x+w, :])

            image.save("%s/%d_%d.jpg" % (output_dir, i, j))

            print "saved %s - %d time(s)" % (image_path, j+1)
