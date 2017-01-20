import tensorflow as tf

from data_reader import DataReader


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


def conv_max_pooling_layer(input, shape, name, max_summary_images=0):
    with tf.name_scope(name):
        # Initialize variables
        W_conv = weight_variable(shape)
        b_conv = bias_variable([shape[3]])

        # Make convolutional operation
        h_conv = conv2d(input, W_conv)

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

        # Pooling
        h_pool = max_pool_2x2(h_conv)

        return h_pool


def get_classmap(label, conv3, name, batch_size=None):

    conv3_resized = tf.image.resize_bilinear(conv3, [10, 10])

    with tf.variable_scope("gap", reuse=True):
        label_w = tf.gather(tf.transpose(tf.get_variable("W")), label)
        label_w = tf.reshape(label_w, [64, 1])

    # Unpack images to list
    conv3_resized_unpacks = tf.unpack(conv3_resized, batch_size)

    classmaps = []

    for unpack in conv3_resized_unpacks:
        unpack = tf.reshape(unpack, [10 * 10, 64])

        classmap = tf.batch_matmul(unpack, label_w)
        classmap = tf.reshape(classmap, [10, 10])
        classmap = tf.expand_dims(classmap, axis=2)

        classmaps.append(classmap)

    pack = tf.pack(classmaps)

    tf.summary.image("%s_classmap" % name, pack, batch_size)


def net(iterations, ckpt_dir, ckpt_file, batch_size):
    sess = tf.InteractiveSession()

    with tf.name_scope("input"):
        # Inputs x -> image data, y_ -> label
        x = tf.placeholder(tf.float32, [None, 80, 80, 1])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

    # Reshape image data into 4D tensor
    x_image = tf.reshape(x, [-1, 80, 80, 1])

    # Add input images to summary
    tf.summary.image('input', x_image, max_outputs=batch_size)

    # First convolution and pooling (5x5 kernel, 16 filters)
    conv1_pool = conv_max_pooling_layer(x_image, [5, 5, 1, 16],
                                        "conv1", 16)

    # Second convolution and pooling (5x5 kernel, 32 filters)
    conv2_pool = conv_max_pooling_layer(conv1_pool, [5, 5, 16, 32],
                                        "conv2", 32)

    # Third convolution and pooling (5x5 kernel, 64 filters)
    conv3_pool = conv_max_pooling_layer(conv2_pool, [5, 5, 32, 64],
                                        "conv3", 64)

    gap = tf.reduce_mean(conv3_pool, [1, 2])

    with tf.variable_scope("gap"):
        gap_w = tf.get_variable(
            "W",
            shape=[64, 2],
            initializer=tf.random_normal_initializer(0., 0.01)
        )

    y = tf.matmul(gap, gap_w)

    get_classmap(0, conv3_pool, "fin", batch_size)
    get_classmap(1, conv3_pool, "no_fin", batch_size)

    """
    with tf.name_scope("fully_connected"):
        # Flatten convolution for fully connected layer
        conv3_pool_flat = tf.reshape(conv3_pool, [-1, 10 * 10 * 64])

        # Fully connected layer weight and bias variables
        w_fullyconn = weight_variable([10 * 10 * 64, 512])
        b_fullyconn = bias_variable([512])

        # Fully connected layer
        fully_connected = tf.nn.relu(
            tf.matmul(conv3_pool_flat, w_fullyconn) + b_fullyconn)

        # Dropout 80% keep
        dropout = tf.nn.dropout(fully_connected, 0.8)

    with tf.name_scope("readout"):
        # Readout layer, map to output classes
        w_readout = weight_variable([512, 2])
        b_readout = bias_variable([2])
        y = tf.matmul(dropout, w_readout) + b_readout
    """

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

    # Setup summary
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(
        "D:\\Stuff\\Faks\\BIOINF\\Projekt\\fin_or_not\\tensorflow_log",
        sess.graph
    )

    # Get data reader
    data_reader = DataReader(
        "D:\\Stuff\\Faks\\BIOINF\\Projekt\\localization_data\\training\\",
        batch_size=batch_size,
        file_names=False
    )

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt:
        saver.restore(sess, ckpt)
        print("Model loaded from file: %s" % ckpt)

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Cumulative moving average
    cma = 0
    n = 0

    for i in range(iterations):
        # Get next batch of images and labels
        batch = data_reader.next()

        # Generate intermediate report
        if i % 10 == 0:
            # Get and print accuracy
            train_accuracy = accuracy.eval(
                feed_dict={x: batch[0], y_: batch[1]}
            )
            print("Training step %d, training accuracy %g" %
                  (i, train_accuracy))

            # Calculate and print Cumulative moving average
            n += 1
            cma += (train_accuracy - cma) / n
            print("Cumulative moving average: %f" % cma)

            # Generate summary
            summary = sess.run(
                merged, feed_dict={x: batch[0], y_: batch[1]}
            )

            run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            writer.add_run_metadata(run_metadata, 'step%03d' % i)
            writer.add_summary(summary, i)
            print('Adding run metadata for', i)

        if i % 100 == 0:
            # Save the variables to disk
            save_path = saver.save(sess, ckpt_dir + ckpt_file)
            print("Model saved in file: %s" % save_path)

        # Train step
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    save_path = saver.save(sess, ckpt_dir + ckpt_file)
    print("Model saved in file: %s" % save_path)

    # Get data reader
    data_reader = DataReader(
        "D:\\Stuff\\Faks\\BIOINF\\Projekt\\localization_data\\testing\\",
        batch_size=1,
        file_names=True
    )

    for i in range(30):
        # Get next batch of images and labels
        batch = data_reader.next()

        prediction = sess.run(y, feed_dict={x: batch[0], y_: batch[1]})
        prediction = tf.argmax(prediction, 1).eval()
        print("Prediction for %s is %s (0 - fin, 1 - no fin):"
              % (batch[2], prediction))

        # Get and print accuracy
        test_accuracy = accuracy.eval(
            feed_dict={x: batch[0], y_: batch[1]}
        )
        print("Testing step %d, testing accuracy %g" % (i, test_accuracy))

net(
    3000,
    "D:\\Stuff\\Faks\\BIOINF\\Projekt\\fin_or_not\\",
    "model.ckpt",
    20
)