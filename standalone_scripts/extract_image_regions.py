import glob
import cv2
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import shuffle

try:
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    print 'Input', input_directory
    print 'Output', output_directory
except:
    print "Usage example \n\t %s ../example/ output_dir/" % os.path.basename(sys.argv[0])
    exit(-1)

start_pos = (0, 0)
end_pos = (0, 0)
fin_rectangles = []
nofin_rectangles = []
selecting_nofin_area = False

def mouse_press_callback(event):
    global start_pos
    print 'press'
    x, y = map(int, (event.xdata, event.ydata))
    start_pos = (x, y)

active_patches = []
txt = None
def key_press_callback(event):
    global txt, fin_rectangles, nofin_rectangles, selecting_nofin_area, active_patches
    if event.key == 'q':
        quit()
    elif event.key == 'n':
        selecting_nofin_area = not selecting_nofin_area
        color = 'red' if selecting_nofin_area else 'green'
        text = 'NOFIN' if selecting_nofin_area else 'FIN'
        if txt is not None:
            txt.remove()

        txt = axes.text(20, 180, 'MODE: %s' % text, color=color)
        image_figure.canvas.draw()
    elif event.key == 'c':
        nofin_rectangles = []
        fin_rectangles = []
        for patch in active_patches:
            patch.remove()
        active_patches = []
        image_figure.canvas.draw()
    elif event.key == 'enter':
        def save_rect(rect, fin_or_nofin):
            global saved_images

            start, end = rect
            start = (int(start[0] / image_ratio), int(start[1] / image_ratio))
            end = (int(end[0] / image_ratio), int(end[1] / image_ratio))
            x1, y1 = min(start[0], end[0]), min(start[1], end[1])
            x2, y2 = max(start[0], end[0]), max(start[1], end[1])

            print 'From to', x1, y1, x2, y2
            print 'Image orig sh', image_original.shape
            area = image_original[y1:y2, x1:x2]
            resized_area = cv2.resize(area, (512, 512))
            print 'writing to', os.path.join(output_directory, '%04d%s.jpg' % (saved_images, fin_or_nofin))
            cv2.imwrite(os.path.join(output_directory, '%04d%s.jpg' % (saved_images, fin_or_nofin)), resized_area)
            saved_images += 1

        for rect in fin_rectangles:
            save_rect(rect, 'fin')
        for rect in nofin_rectangles:
            save_rect(rect, 'nofin')

        fin_rectangles = []
        nofin_rectangles = []        
        active_patches = []

        plt.close()
        

def mouse_release_callback(event):
    global start_pos, end_pos, fin_rectangles, nofin_rectangles, selecting_nofin_area
    print 'release'
    x, y = map(int, (event.xdata, event.ydata))
    end_pos = (x, y)
    x1, x2 = min(start_pos[0], end_pos[0]), max(start_pos[0], end_pos[0])
    y1, y2 = min(start_pos[1], end_pos[1]), max(start_pos[1], end_pos[1])
    w, h = x2 - x1, y2 - y1
    if selecting_nofin_area:
        patch = patches.Rectangle((x1, y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
        axes.add_patch(patch)
        active_patches.append(patch)
        nofin_rectangles.append((start_pos, end_pos))
    else:
        patch = patches.Rectangle((x1, y1),w,h,linewidth=1,edgecolor='g',facecolor='none')
        axes.add_patch(patch)
        active_patches.append(patch)
        fin_rectangles.append((start_pos, end_pos))
    image_figure.canvas.draw()

#cv2.setMouseCallback('Image', mouse_callback)
saved_images = 0

all_image_paths = list(glob.glob(os.path.join(input_directory, '*.jpg')))
shuffle(all_image_paths)
for image_full_path in all_image_paths:
    print 'Opening image', image_full_path
    active_patches = []

    image_original = cv2.imread(image_full_path)
    image_h, image_w = image_original.shape[:2]
    image_w_small = 800
    image_ratio = float(image_w_small) / image_w
    image_h_small = int(image_ratio * image_h)

    image_scaled = cv2.resize(image_original, (image_w_small, image_h_small))

    cv2.putText(image_scaled, 'c - clear image', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(image_scaled, 'click n drag - draw fin', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(image_scaled, 'n - toggle fin/nofin mode', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(image_scaled, 'enter - next image', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(image_scaled, 'q - quit', (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    for rect in fin_rectangles:
        cv2.rectangle(image_scaled, rect[0], rect[1], (0, 255, 0))
    for rect in nofin_rectangles:
        cv2.rectangle(image_scaled, rect[0], rect[1], (0, 0, 255))

    image_figure, axes = plt.subplots(1)
    image_figure.canvas.mpl_connect('button_press_event', mouse_press_callback)
    image_figure.canvas.mpl_connect('button_release_event', mouse_release_callback)
    image_figure.canvas.mpl_connect('key_press_event', key_press_callback)

    axes.imshow(cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB))

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.show()

        