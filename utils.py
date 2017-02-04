import cv2


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
