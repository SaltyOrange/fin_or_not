from localization_pipeline.pipeline import Pipeline

import sys
import os
import glob
import cv2

import matplotlib.pyplot as plt

try:
    localization_dir = sys.argv[1]
    classification_dir = sys.argv[2]
    test_images_directory = sys.argv[3]
    test_rect_directory = sys.argv[4]
    output_dir = sys.argv[5]
except:
    print """
        Usage:
        scripy.py localization_dir classification_dir test_images_directory test_rect_directory output_dir
    """
    exit(-1)


def rects_overlap_ratio(rect1, rect2):
    """ 
    Returns ratio:
        (r1 intersection r2)/(r1 union r2)
    """
    l1, t1, r1, b1 = rect1[0], rect1[1], rect1[0] + rect1[2], rect1[1] + rect1[3]
    l2, t2, r2, b2 = rect2[0], rect2[1], rect2[0] + rect2[2], rect2[1] + rect2[3]

    # coordinates of intersection
    l3, t3, r3, b3 = max(l1, l2), max(t1, t2), min(r1, r2), min(b1, b2)

    # check if intersecton exists
    if l3 < r3 and t3 < b3:
        area1 = rect1[2] * rect1[3]
        area2 = rect2[2] * rect2[3]
        area3 = (r3 - l3) * (b3 - t3)
        area_union = area1 + area2 - area3
        
        ratio_intersection_to_union = float(area3) / area_union
        return ratio_intersection_to_union
    return 0.0

pipeline = Pipeline(localization_dir, classification_dir)

num_positive_hasfin_predictions = 0
num_negative_hasfin_predictions = 0
num_images = 0

num_positive_rects = 0 # predicted rects that overlap with real rects
num_negative_rects = 0 # predicted rects that don't overlap with real rects
num_real_rects = 0 # total real rects in all images

num_total_predicted_rects = 0

for image_path in glob.glob(os.path.join(test_images_directory, '*.jpg')):
    #image = cv2.imread(image_path)
    print 'Processing image', image_path

    num_images += 1
    
    image_name = os.path.basename(image_path)
    rect_path = os.path.join(test_rect_directory, image_name[:-3] + 'rects')
    image_contains_fin = 'no_fin' not in image_name

    # Check if .rects file exist
    if os.path.isfile(rect_path):
        with open(rect_path, 'r') as rect_file:
            file_lines = rect_file.readlines()
            real_rects = [map(int, line.strip().split()) for line in file_lines]
    else:
        real_rects = []

    predicted_contains_fin, predicted_rects, classmap = pipeline.predict(image_path)

    print 'Predicted contains fin:', predicted_contains_fin, 'real:', image_contains_fin
    print 'Prediction correct:', predicted_contains_fin == image_contains_fin

    # Write classmap to file
    cv2.imwrite(os.path.join(output_dir, 'cm_' + image_name), classmap)

    if predicted_contains_fin == image_contains_fin:
        num_positive_hasfin_predictions += 1
    else:
        num_negative_hasfin_predictions += 1

    num_real_rects += len(real_rects)

    if predicted_contains_fin == False:
        continue
    
    num_total_predicted_rects += len(predicted_rects)


    # Draw real recangles
    image = cv2.imread(image_path)
    for r in real_rects:
        x, y, w, h = r
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0))

    for r in predicted_rects:
        x, y, w, h = r
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255))
    
    # Now for each predicted rect, check if it is a good prediction
    for predicted_rect in predicted_rects:

        overlaps_with_real = False
        if len(real_rects) > 0:
            max_overlaping_rect = max(real_rects, key=lambda rect: rects_overlap_ratio(predicted_rect, rect))

            overlap_ratio = rects_overlap_ratio(predicted_rect, max_overlaping_rect)
            if overlap_ratio > 0.0:
                overlaps_with_real = True
                real_rects.remove(max_overlaping_rect)
            

        """
        for real_rect in real_rects:
            if rects_overlap(predicted_rect, real_rect):
                overlaps_with_real = True
                real_rects.remove(real_rect)
                break
        else:
            overlaps_with_real = False
        """
        if overlaps_with_real:
            num_positive_rects += 1
        else:
            num_negative_rects += 1

    cv2.imwrite(os.path.join(output_dir, image_name), image)
print 'Number of images correctly classified as containing fin:', num_positive_hasfin_predictions
print 'Number of images incorrectly classified as containing fin:', num_negative_hasfin_predictions
print 'Number of images total', num_images

print 'Number of correctly predicted fin rectangles', num_positive_rects
print 'Number of incorrectly predicted fin rectangles', num_negative_rects
print 'Number of real rects', num_real_rects
print 'Number of total predicted rects:', num_total_predicted_rects
