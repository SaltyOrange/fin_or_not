import tensorflow as tf
import numpy as np

from PIL import Image


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

    image = Image.open(image_path).convert("L")
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=3)

    prediction = sess.run(y, feed_dict={x: image_array})
    print(tf.argmax(prediction, 1).eval())

predict(
    "D:\\Stuff\\Faks\\BIOINF\\Projekt\\localization_data\\testing\\IMG_87_2.jpg",
    "D:\\Stuff\\Faks\\BIOINF\\Projekt\\fin_or_not\\",
    "model.ckpt.meta"
)