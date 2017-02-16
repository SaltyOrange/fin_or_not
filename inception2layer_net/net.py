import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, padding, strides):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)


def image_summary(tensor, img_height, img_width,
                  img_channels, name, max_outputs):

    # Get the first image from this batch
    summary_image = tf.slice(tensor, (0, 0, 0, 0), (1, -1, -1, -1),
                             name='slice_first_input')
    # Reshape into 3D tensor
    summary_image = tf.reshape(summary_image,
                               (img_height, img_width, img_channels))

    # Reorder so the channels are in the first dimension, x and y follow
    summary_image = tf.transpose(summary_image, (2, 0, 1))
    # Bring into shape expected by image_summary
    summary_image = tf.reshape(summary_image,
                               (-1, img_height, img_width, 1))

    # Add to summary
    tf.summary.image(name, summary_image, max_outputs)


def conv_layer(input, shape, name, padding,
               strides=None, max_summary_images=0):

    if not strides:
        strides = [1, 1, 1, 1]

    with tf.name_scope(name):
        # Initialize variables
        W_conv = weight_variable(shape)
        b_conv = bias_variable([shape[3]])

        l2_loss = tf.nn.l2_loss(W_conv)
        tf.add_to_collection('l2_losses', l2_loss)

        # Make convolutional operation
        h_conv = conv2d(input, W_conv, padding, strides=strides)

        # Add images to summary
        if max_summary_images > 0:
            image_summary(tf.transpose(W_conv, (2, 0, 1, 3)),
                          shape[0], shape[1], shape[3],
                          "%s_filters" % name, max_summary_images)

            image_summary(h_conv, input.get_shape()[1].value,
                          input.get_shape()[2].value,
                          shape[3], name, max_summary_images)

        # RELU operation
        h_conv = tf.nn.relu(h_conv + b_conv)

        return h_conv


def get_classmap(label, conv3, name, batch_size=1):

    conv3_resized = tf.image.resize_bilinear(conv3, [64, 64])

    with tf.variable_scope("gap", reuse=True):
        label_w = tf.gather(tf.transpose(tf.get_variable("W")), label)
        label_w = tf.reshape(label_w, [1024, 1])

    # Unpack images to list
    conv3_resized_unpacks = tf.unpack(conv3_resized, batch_size)

    classmaps = []

    for unpack in conv3_resized_unpacks:
        unpack = tf.reshape(unpack, [64 * 64, 1024])

        classmap = tf.matmul(unpack, label_w)
        classmap = tf.reshape(classmap, [64, 64])
        classmap = tf.expand_dims(classmap, axis=2)

        classmaps.append(classmap)

    pack = tf.pack(classmaps)

    tf.summary.image("%s_classmap" % name, pack, batch_size)


def inception_layer(input, input_size, param1, param2, param3, param4, param5,
                    param6, name):

    # inception layer start

    # 1st branch
    inception_1x1 = conv_layer(input, [1, 1, input_size, param1],
                               padding="SAME",
                               strides=[1, 1, 1, 1],
                               name="%s_1x1" % name)

    # 2nd branch
    inception_3x3_reduce = conv_layer(input, [1, 1, input_size, param2],
                                      padding="SAME",
                                      strides=[1, 1, 1, 1],
                                      name="%s_3x3_reduce" % name)

    inception_3x3 = conv_layer(inception_3x3_reduce, [3, 3, param2, param3],
                               padding="SAME",
                               strides=[1, 1, 1, 1],
                               name="%s_3x3" % name)

    # 3rd branch
    inception_5x5_reduce = conv_layer(input, [1, 1, input_size, param4],
                                      padding="SAME",
                                      strides=[1, 1, 1, 1],
                                      name="%s_5x5_reduce" % name)

    inception_5x5 = conv_layer(inception_5x5_reduce, [5, 5, param4, param5],
                               padding="SAME",
                               strides=[1, 1, 1, 1],
                               name="%s_5x5" % name)

    # 4th branch
    inception_pool = tf.nn.max_pool(input, ksize=[1, 3, 3, 1],
                                    strides=[1, 1, 1, 1], padding='SAME',
                                    name="%s_pool" % name)

    inception_pool_proj = conv_layer(inception_pool,
                                     [1, 1, input_size, param6],
                                     padding="SAME",
                                     strides=[1, 1, 1, 1],
                                     name="%s_pool_proj" % name)
    # inception layer end

    return tf.concat(3, [inception_1x1,
                         inception_3x3,
                         inception_5x5,
                         inception_pool_proj],
                     name="%s_output" % name)


def inference(batch_size):

    with tf.name_scope("input"):
        # Inputs x -> image data, y_ -> label
        x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        y_true = tf.placeholder(tf.float32, shape=[None, 2])

    # Add input images to summary
    tf.summary.image('input', x, max_outputs=batch_size)

    conv1_7x7_s2 = conv_layer(x, [7, 7, 3, 64], padding="SAME",
                              strides=[1, 2, 2, 1],
                              name="conv1/7x7_s2")

    pool1_3x3_s2 = tf.nn.max_pool(conv1_7x7_s2, ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1], padding='SAME',
                                  name="pool1_3x3_s2")

    # TODO: Change LRN parameters
    pool1_norm1 = tf.nn.local_response_normalization(pool1_3x3_s2)

    conv2_3x3_reduce = conv_layer(pool1_norm1, [1, 1, 64, 64], padding="VALID",
                                  strides=[1, 1, 1, 1],
                                  name="conv2_3x3_reduce")

    conv2_3x3 = conv_layer(conv2_3x3_reduce, [3, 3, 64, 192], padding="SAME",
                           strides=[1, 1, 1, 1],
                           name="conv2_3x3")

    # TODO: Change LRN parameters
    conv2_norm2 = tf.nn.local_response_normalization(conv2_3x3)

    pool2_3x3_s2 = tf.nn.max_pool(conv2_norm2, ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1], padding='SAME',
                                  name="pool2_3x3_s2")

    inception_3a_output = inception_layer(pool2_3x3_s2, 192, 64, 96, 128, 16,
                                          32, 32, "inception_3a")

    inception_3b_output = inception_layer(inception_3a_output, 256, 128, 128,
                                          192, 32, 96, 64, "inception_3b")

    final_conv = conv_layer(inception_3b_output, [3, 3, 480, 1024],
                            padding="SAME",
                            strides=[1, 1, 3, 1],
                            name="final_conv")

    gap = tf.reduce_mean(final_conv, [1, 2], name="gap")

    with tf.variable_scope("gap"):
        gap_w = tf.get_variable(
            "W",
            shape=[1024, 2],
            initializer=tf.random_normal_initializer(0., 0.01)
        )

    y = tf.matmul(gap, gap_w, name="y")

    get_classmap(0, final_conv, "fin", batch_size)
    get_classmap(1, final_conv, "no_fin", batch_size)

    with tf.name_scope("accuracy"):
        # Accuracy measure
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope("optimizer"):
        # Cross entropy loss function
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(y, y_true))

        weight_decay_constant = 0.0005
        loss = cross_entropy + tf.add_n(
            tf.get_collection("l2_losses")) * weight_decay_constant

        tf.summary.scalar('loss', loss)

        # Training step optimizer
        train_step = tf.train.AdamOptimizer(0.0004).minimize(loss)

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

    return x, y_true, y, gap_w, final_conv, train_step, accuracy, saver