import tensorflow as tf
import numpy as np

from PIL import Image


def get_classmap(label, conv3, w):

    conv3_resized = tf.image.resize_bilinear(conv3, [64, 64])

    label_w = tf.gather(tf.transpose(w), label)
    label_w = tf.reshape(label_w, [512, 1])
    conv3_resized = tf.reshape(conv3_resized, [64 * 64, 512])

    classmap = tf.matmul(conv3_resized, label_w)
    classmap = tf.reshape(classmap, [64, 64])

    return classmap


def predict(image_path, model_dir, model_meta):
    sess = tf.InteractiveSession()

    saver = tf.train.import_meta_graph(model_dir + model_meta)

    ckpt = tf.train.latest_checkpoint(model_dir)
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

    image = Image.open(image_path)
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)
    # image_array = np.expand_dims(image_array, axis=3)

    prediction, conv3, w = sess.run([y, conv3, w], feed_dict={x: image_array})
    print(tf.argmax(prediction, 1).eval())

    classmap_image = Image.fromarray(get_classmap(0, conv3, w).eval())

    classmap_image = classmap_image.resize(image.size)
    classmap_image = classmap_image.convert(mode="RGBA")

    image = image.convert(mode="RGBA")

    image = Image.blend(image, classmap_image, 0.5)

    image.show()

predict(
    "D:\\Stuff\\Faks\\BIOINF\\Projekt\\localization_data\\weakly_color\\IMG_66_0.jpg",
    "D:\\Stuff\\Faks\\BIOINF\\Projekt\\fin_or_not\\",
    "model.ckpt.meta"
)