import tensorflow as tf
import numpy as np
import sys

from PIL import Image


def files_in_directory(directory, allowed_extensions):
    import os
    # print 'Reading all files in directory:', directory
    
    for file_or_dir in os.listdir(directory):
        full_path_to_file_or_dir = os.path.join(directory, file_or_dir)

        if os.path.isfile(full_path_to_file_or_dir) == False:
            continue
        
        file_extension = full_path_to_file_or_dir.lower().split('.')[-1]
        if file_extension in allowed_extensions:
            yield full_path_to_file_or_dir


def get_classmap(label, conv3, w):

    conv3_resized = tf.image.resize_bilinear(conv3, [64, 64])

    label_w = tf.gather(tf.transpose(w), label)
    label_w = tf.reshape(label_w, [128, 1])
    conv3_resized = tf.reshape(conv3_resized, [64 * 64, 128])

    classmap = tf.matmul(conv3_resized, label_w)
    classmap = tf.reshape(classmap, [64, 64])

    return classmap


def get_session(model_dir):
    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph(model_dir + "model.ckpt")
    # print(saver.last_checkpoints)

    # ckpt = tf.train.latest_checkpoint(model_dir)
    ckpt = model_dir + "model.ckpt"
    if ckpt:
        saver.restore(sess, ckpt)
        print("Model loaded from file: %s" % ckpt)
    else:
        print("No model checkpoint found")
        exit(-1)

    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]

    conv3 = tf.get_collection("conv3")[0]
    w = tf.get_collection("gap_w")[0]

    return sess, x, y, conv3, w


def predict(image_path, sess, image_save_dir, image_save_name, save_type=0):
    """
    Predicts class for given image ans saves selected images
    (classmap, blended classmap and image or both)
    :param image_path: input image
    :param sess: tensorflow session used for prediction
    :param image_save_dir: for image saving
    :param image_save_name: for image saving
    :param save_type: what to save (blend, classmap or both) (0, 1, 2)
    """

    # print "Opening image", image_path
    image = Image.open(image_path)
    image = image.resize((512, 512), Image.ANTIALIAS)

    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)
    # image_array = np.expand_dims(image_array, axis=3)

    prediction, conv3_array, w_array = sess.run([y, conv3, w],
                                                feed_dict={x: image_array})
    prediction = tf.argmax(prediction, 1).eval()

    classmap_image = Image.fromarray(
        get_classmap(0, conv3_array, w_array).eval())

    if save_type == 1 or save_type == 2:
        classmap_image.save("%s/%s_class_%s_classmap.jpg" %
                            (image_save_dir, image_save_name, str(prediction)))

    if save_type == 0 or save_type == 2:
        classmap_image = classmap_image.resize(image.size)
        classmap_image = classmap_image.convert(mode="RGBA")

        image = image.convert(mode="RGBA")
        image = Image.blend(image, classmap_image, 0.5)

        image.save("%s/%s_class_%s_blend.jpg" %
                   (image_save_dir, image_save_name, str(prediction)))


if len(sys.argv) < 4:
    sys.exit("Usage: script.py model_dir image_load_dir image_save_dir")

session, x, y, conv3, w = get_session(sys.argv[1])

image_directory = sys.argv[2]

for i, image_path in enumerate(files_in_directory(image_directory, "jpg")):
    predict(image_path, session, sys.argv[3], str(i), save_type=1)
