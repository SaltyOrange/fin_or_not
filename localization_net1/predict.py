import tensorflow as tf
import numpy as np

from PIL import Image


class Predictor:

    def __init__(self, model_dir):
        self.session, self.x, self.y, self.conv3, self.w = \
            self.get_session(model_dir)

    def close_session(self):
        self.session.close()

    def get_session(self, model_dir):
        sess = tf.Session()
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

        conv3 = tf.get_collection("conv3")[0]
        w = tf.get_collection("gap_w")[0]

        return sess, x, y, conv3, w

    def get_classmap(self, label, conv3, w):

        conv3_resized = tf.image.resize_bilinear(conv3, [64, 64])

        label_w = tf.gather(tf.transpose(w), label)
        label_w = tf.reshape(label_w, [128, 1])
        conv3_resized = tf.reshape(conv3_resized, [64 * 64, 128])

        classmap = tf.matmul(conv3_resized, label_w)
        classmap = tf.reshape(classmap, [64, 64])

        return classmap

    def predict(self, image_path, return_type=0):
        """
        Predicts class for given image ans saves selected images
        (classmap, blended classmap and image or both)
        :param image_path: input image
        :param return_type: what to return (blend, classmap or both) (0, 1, 2)
        """

        # print "Opening image", image_path
        image = Image.open(image_path)
        image = image.resize((512, 512), Image.ANTIALIAS)

        image_array = np.asarray(image)
        image_array = np.expand_dims(image_array, axis=0)
        # image_array = np.expand_dims(image_array, axis=3)

        prediction, conv3_array, w_array = self.session.run(
            [self.y, self.conv3, self.w],
            feed_dict={self.x: image_array}
        )
        prediction = tf.argmax(prediction, 1).eval()

        classmap_image = Image.fromarray(
            self.get_classmap(0, conv3_array, w_array).eval())

        return_list = []

        if return_type == 0 or return_type == 2:
            classmap_image = classmap_image.resize(image.size)
            classmap_image = classmap_image.convert(mode="RGBA")

            image = image.convert(mode="RGBA")
            image = Image.blend(image, classmap_image, 0.5)

            return_list.append(image)

        if return_type == 1 or return_type == 2:
            classmap_image = classmap_image.convert("RGB")
            return_list.append(classmap_image)

        return prediction, return_list