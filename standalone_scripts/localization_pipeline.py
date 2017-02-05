import sys

import numpy
import cv2
from PIL import Image

from localization_net1.predict import Predictor as LocalizationPredictor
from classification_net.predict import Predictor as ClassificationPredictor
from utils import get_blob_bounding_boxes


if len(sys.argv) < 3:
    sys.exit("Usage: script.py localization_model_dir classification_model_dir"
             " image_path")

# Get predictors
localization_predictor = LocalizationPredictor(sys.argv[1])
classification_predictor = ClassificationPredictor(sys.argv[2])

# Get image path
image_path = sys.argv[3]

# Get prediction and classmap for image
prediction, images = \
    localization_predictor.predict(image_path, return_type=1)

classmap = images[0].convert("L")
classmap = numpy.array(classmap)

# Get bounding boxes for blobs in classmap
bounding_boxes = get_blob_bounding_boxes(classmap)

# Translate bounding boxes into original picture coordinate system
original_image = cv2.imread(image_path)

classmap_height, classmap_width = classmap.shape
original_image_height, original_image_width, _ = original_image.shape
width_ratio = original_image_width/classmap_width
height_ratio = original_image_height/classmap_height

for box in bounding_boxes:
    x, y, w, h = box
    # Bounding box coordinate translation
    x *= width_ratio
    w *= width_ratio
    y *= height_ratio
    h *= height_ratio

    # Check if really a fin using CNN
    prediction = classification_predictor.predict(
        Image.fromarray(original_image[y-1:y+h-1, x-1:x+h-1, :]))

    # If it really is a fin draw a rectange
    if prediction == [0]:
        cv2.rectangle(original_image, (x, y), (x + w, y + h),
                      (128, 128, 128), 0)

# Show image
cv2.imshow("Image with bounding boxes", original_image)
cv2.waitKey()
cv2.destroyAllWindows()

