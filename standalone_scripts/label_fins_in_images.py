import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpp

import sys
import os
import glob

def usage():
    print 'USAGE:'
    print '    script.py input_directory {skip|overwrite}'
    print ''
    print '    For each jpg file in input_directory you can label a rectangle. 
    print '    When you press enter the rectangle is saved next to the file with .rects extension'
    print '    You can clear drawn rectangles by pressing "c"'
    print '    You can overwrite existing rectangle files or skip images with existing .rects file, depending on passed parameter'
    exit(-1)

try:
    input_directory = sys.argv[1]
    if sys.argv[2] == 'skip':
        skip_existing_files = True
    elif sys.argv[2] == 'overwrite':
        skip_existing_files = False
    else:
        usage()
except:
    usage()


def onmousedown(event):
    global rect_p1
    x, y = int(event.xdata), int(event.ydata)
    rect_p1 = (x, y)


def onmouseup(event):
    global rect_p1, image_rects
    x1, y1 = rect_p1
    x2, y2 = int(event.xdata), int(event.ydata)
    x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    
    x, y, w, h = (x1, y1, (x2 - x1), (y2 - y1))

    if (w < 3 or h < 3): return

    rect = mpp.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    image_rects.append(rect)
    image_plot.axes.add_patch(rect)
    image_plot.figure.canvas.draw()


def onkeypress(event):
    global image_rects
    if event.key == 'enter':
        rects = [(r.get_x(), r.get_y(), r.get_width(), r.get_height()) for r in image_rects]
        rects = [map(lambda x: str(int(x)), rect) for rect in rects]
        rect_strings = [' '.join(rect) + '\n' for rect in rects]

        if len(image_rects) > 0:
            print 'Writing %d rects to %s' % (len(image_rects), image_rect_file)    
            with open(image_rect_file, 'w') as outfile:
                outfile.writelines(rect_strings)
        
        plt.close()
    if event.key == 'c':
        for rect in image_rects:
            rect.remove()
        image_rects = []
        image_plot.figure.canvas.draw()
    

for image_path in glob.glob(os.path.join(input_directory, '*.jpg')):
    print 'Opening', image_path
    image_rects = []
    image_rect_file = image_path[:-3] + 'rects'
    
    if os.path.isfile(image_rect_file) and skip_existing_files:
        continue # skip this image
    
    image = mpimg.imread(image_path)

    # Create figure and hook callbacks
    image_plot = plt.imshow(image)
    image_plot.figure.canvas.mpl_connect('button_press_event', onmousedown)
    image_plot.figure.canvas.mpl_connect('button_release_event', onmouseup)
    image_plot.figure.canvas.mpl_connect('key_press_event', onkeypress)

    plt.show()
    

    
