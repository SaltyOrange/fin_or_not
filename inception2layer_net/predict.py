import tensorflow as tf
import numpy as np

from PIL import Image


class Predictor:

    def __init__(self, model_dir):
        self.session, self.x, self.y, self.conv3, self.w = \
            self.get_session(model_dir)

    def close_session(self):
        self.session.close()
        tf.reset_default_graph()

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

        pool5_7x7_s1 = tf.get_collection("pool5_7x7_s1")[0]
        w = tf.get_collection("gap_w")[0]

        return sess, x, y, pool5_7x7_s1, w

    def get_classmap(self, label, pool5_7x7_s1, w):

        pool5_7x7_s1_resized = tf.image.resize_bilinear(pool5_7x7_s1, [64, 64])

        label_w = tf.gather(tf.transpose(w), label)
        label_w = tf.reshape(label_w, [1024, 1])
        pool5_7x7_s1_resized = tf.reshape(pool5_7x7_s1_resized,
                                          [64 * 64, 1024])

        classmap = tf.matmul(pool5_7x7_s1_resized, label_w)
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
        image = image.resize((224, 224), Image.ANTIALIAS)

        image_array = np.asarray(image)
        image_array = np.expand_dims(image_array, axis=0)
        # image_array = np.expand_dims(image_array, axis=3)

        prediction, pool5_7x7_s1_array, w_array = self.session.run(
            [self.y, self.conv3, self.w],
            feed_dict={self.x: image_array}
        )
        prediction = tf.argmax(prediction, 1).eval(session=self.session)

        classmap_image = Image.fromarray(
            self.get_classmap(0, pool5_7x7_s1_array, w_array).eval(
                session=self.session))

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