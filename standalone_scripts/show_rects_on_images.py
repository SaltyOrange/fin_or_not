import os
import glob
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as plimg

images_dir = sys.argv[1]
rects_dir = sys.argv[2]

for image_path in glob.glob(os.path.join(images_dir, '*.jpg')):
    print 'Opening', image_path
    
    image_name = os.path.basename(image_path)
    no_fin = 'no_fin' in image_name
    if no_fin:
        continue

    rects_path = os.path.join(rects_dir, image_name[:-3] + 'rects')    
    with open(rects_path, 'r') as rect_file:
        rects = [map(int, line.strip().split()) for line in rect_file.readlines()]
        
    img = plimg.imread(image_path)
    
    for rect in rects:
        x, y, w, h = rect    
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0))

    plt.imshow(img)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

