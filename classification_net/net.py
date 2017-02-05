import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def image_summary(tensor, img_height, img_width,
                  img_channels, name, max_outputs, rgb=False):

    if not rgb:
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
    else:
        # Reorder so the channels are in the first dimension, x and y follow
        summary_image = tf.transpose(tensor, (3, 0, 1, 2))

        # Bring into shape expected by image_summary
        summary_image = tf.reshape(summary_image,
                                   (-1, img_height, img_width, 3))

    # Add to summary
    tf.summary.image(name, summary_image, max_outputs)


def conv_max_pooling_layer(input, shape, name, max_summary_images=0):
    with tf.name_scope(name):
        # Initialize variables
        W_conv = weight_variable(shape)
        b_conv = bias_variable([shape[3]])

        # Make convolutional operation
        h_conv = conv2d(input, W_conv)

        # Add images to summary
        if max_summary_images > 0:

            if shape[2] == 3:
                image_summary(tf.transpose(W_conv, (2, 0, 1, 3)),
                              shape[0], shape[1], shape[3],
                              "%s_filters" % name, max_summary_images, True)
            else:
                image_summary(tf.transpose(W_conv, (2, 0, 1, 3)),
                              shape[0], shape[1], shape[3],
                              "%s_filters" % name, max_summary_images)

            image_summary(h_conv, h_conv.get_shape()[1].value,
                          h_conv.get_shape()[2].value,
                          h_conv.get_shape()[3].value,
                          name, max_summary_images)

        # RELU operation
        h_conv = tf.nn.relu(h_conv + b_conv)

        # Pooling
        h_pool = max_pool_2x2(h_conv)

        return h_pool


def inference():

    with tf.name_scope("input"):
        # Inputs x -> image data, y_ -> label
        x = tf.placeholder(tf.float32, [None, 80, 80, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

    # Reshape image data into 4D tensor
    x_image = tf.reshape(x, [-1, 80, 80, 3])

    # Add input images to summary
    tf.summary.image('input', x_image, max_outputs=4)

    # First convolution and pooling (5x5 kernel, 8 filters)
    conv1_pool = conv_max_pooling_layer(x_image, [5, 5, 3, 8],
                                        "conv1", 16)

    # Second convolution and pooling (5x5 kernel, 16 filters)
    conv2_pool = conv_max_pooling_layer(conv1_pool, [5, 5, 8, 16],
                                        "conv2", 32)

    # Third convolution and pooling (5x5 kernel, 32 filters)
    conv3_pool = conv_max_pooling_layer(conv2_pool, [5, 5, 16, 32],
                                        "conv3", 64)

    with tf.name_scope("fully_connected"):
        # Flatten convolution for fully connected layer
        conv3_pool_flat = tf.reshape(conv3_pool, [-1, 10 * 10 * 32])

        # Fully connected layer weight and bias variables
        w_fullyconn = weight_variable([10 * 10 * 32, 512])
        b_fullyconn = bias_variable([512])

        # Fully connected layer
        fully_connected = tf.nn.relu(
            tf.matmul(conv3_pool_flat, w_fullyconn) + b_fullyconn)

        # Dropout 50% keep
        dropout = tf.nn.dropout(fully_connected, 0.5)

    with tf.name_scope("readout"):
        # Readout layer, map to output classes
        w_readout = weight_variable([512, 2])
        b_readout = bias_variable([2])
        y = tf.matmul(dropout, w_readout) + b_readout

    with tf.name_scope("accuracy"):
        # Accuracy measure
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope("optimizer"):
        # Cross entropy loss function
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(y, y_))

        tf.summary.scalar('cross_entropy', cross_entropy)

        # Training step optimizer
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    return x, y_, y, train_step, accuracy