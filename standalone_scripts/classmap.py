import sys

from ..predict import Predictor
from ..utils import files_in_directory

if len(sys.argv) < 4:
    sys.exit("Usage: script.py model_dir image_load_dir "
             "image_save_dir save_type \n"
             "save type:what to save (blend, classmap or both) (0, 1, 2)")

predictor = Predictor(sys.argv[1])

image_directory = sys.argv[2]
image_save_dir = sys.argv[3]
save_type = sys.argv[4]

for i, image_path in enumerate(files_in_directory(image_directory, "jpg")):
    prediction, images = \
        predictor.predict(image_path, return_type=save_type)

    if save_type == 0 or save_type == 2:
        images[0].save("%s/%s_class_%s_blend.jpg" %
                       (image_save_dir, str(i), str(prediction)))
        del images[0]

    if save_type == 1 or save_type == 2:
        images[0].save("%s/%s_class_%s_classmap.jpg" %
                       (image_save_dir, str(i), str(prediction)))
        del images[0]
