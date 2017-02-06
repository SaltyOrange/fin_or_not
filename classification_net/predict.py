import tensorflow as tf
import numpy as np

from PIL import Image


class Predictor:

    def __init__(self, model_dir):
        self.session, self.x, self.y = \
            self.get_session(model_dir)

    def get_session(self, model_dir):
        sess = tf.InteractiveSession()
        saver = tf.train.import_meta_graph(model_dir + "ckpt.meta")
        # print(saver.last_checkpoints)

        ckpt = tf.train.latest_checkpoint(model_dir)
        # ckpt = model_dir + "ckpt"
        if ckpt:
            saver.restore(sess, ckpt)
            print("Model loaded from file: %s" % ckpt)
        else:
            print("No model checkpoint found")
            exit(-1)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]

        return sess, x, y

    def predict(self, image):
        """
        Predicts class for given image ans saves selected images
        (classmap, blended classmap and image or both)
        :param image_path: input image
        :param return_type: what to return (blend, classmap or both) (0, 1, 2)
        """

        image = image.resize((80, 80), Image.ANTIALIAS)

        image_array = np.asarray(image)
        image_array = np.expand_dims(image_array, axis=0)
        # image_array = np.expand_dims(image_array, axis=3)

        prediction = self.session.run(
            self.y,
            feed_dict={self.x: image_array}
        )

        return tf.argmax(prediction, 1).eval()