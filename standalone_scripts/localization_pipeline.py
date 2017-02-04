import sys

import cv2

from predict import Predictor
from utils import get_blob_bounding_boxes


if len(sys.argv) < 3:
    sys.exit("Usage: script.py model_dir image_path")

# Get predictor
predictor = Predictor(sys.argv[1])

# Get image path
image_path = sys.argv[2]

# Get prediction and classmap for image
prediction, images = \
    predictor.predict(image_path, return_type=1)

classmap = images[0]

# Get bounding boxes for blobs in classmap
bounding_boxes = get_blob_bounding_boxes(classmap)

# Translate bounding boxes into original picture coordinate system
original_image = cv2.imread(image_path)

classmap_width, classmap_height = classmap.size
original_image_width, original_image_height = original_image.size
width_ratio = original_image_width/classmap_width
height_ratio = original_image_height/classmap_height

for box in bounding_boxes:
    x, y, w, h = box
    x *= width_ratio
    w *= width_ratio
    y *= height_ratio
    h *= height_ratio
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (128, 128, 128), 0)

cv2.imshow("Image with bounding boxes", original_image)
cv2.waitKey()
cv2.destroyAllWindows()