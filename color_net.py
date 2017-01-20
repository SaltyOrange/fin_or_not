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


def net(iterations, ckpt_dir, ckpt_file):
    sess = tf.InteractiveSession()

    with tf.name_scope("input"):
        # Inputs x -> image data, y_ -> label
        x = tf.placeholder(tf.float32, [None, 80, 80, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

    # Reshape image data into 4D tensor
    x_image = tf.reshape(x, [-1, 80, 80, 3])

    # Add input images to summary
    tf.summary.image('input', x_image, max_outputs=4)

    # First convolution and pooling (5x5 kernel, 16 filters)
    conv1_pool = conv_max_pooling_layer(x_image, [5, 5, 3, 16],
                                        "conv1", 16)

    # Second convolution and pooling (5x5 kernel, 32 filters)
    conv2_pool = conv_max_pooling_layer(conv1_pool, [5, 5, 16, 32],
                                        "conv2", 32)

    # Third convolution and pooling (5x5 kernel, 64 filters)
    conv3_pool = conv_max_pooling_layer(conv2_pool, [5, 5, 32, 64],
                                        "conv3", 64)

    with tf.name_scope("fully_connected"):
        # Flatten convolution for fully connected layer
        conv3_pool_flat = tf.reshape(conv3_pool, [-1, 10 * 10 * 64])

        # Fully connected layer weight and bias variables
        w_fullyconn = weight_variable([10 * 10 * 64, 1024])
        b_fullyconn = bias_variable([1024])

        # Fully connected layer
        fully_connected = tf.nn.relu(
            tf.matmul(conv3_pool_flat, w_fullyconn) + b_fullyconn)

        # Dropout 80% keep
        dropout = tf.nn.dropout(fully_connected, 0.8)

    with tf.name_scope("readout"):
        # Readout layer, map to output classes
        w_readout = weight_variable([1024, 2])
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

    # Setup summary
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(
        "D:\\Stuff\\Faks\\BIOINF\\Projekt\\fin_or_not\\tensorflow_log",
        sess.graph
    )

    # Get data reader
    data_reader = DataReader(
        "D:\\Stuff\\Faks\\BIOINF\\Projekt\\localization_data\\color_training\\",
        batch_size=100,
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

    # TODO: Add accuracy based on test data every 10-th step

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

            summary = sess.run(
                merged, feed_dict={x: batch[0], y_: batch[1]}
            )
            writer.add_summary(summary, i)

        if i % 100 == 0:
            run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # Generate summary
            summary = sess.run(
                merged, feed_dict={x: batch[0], y_: batch[1]},
                options=run_options,
                run_metadata=run_metadata
            )

            writer.add_run_metadata(run_metadata, 'step%03d' % i)
            writer.add_summary(summary, i)
            print('Adding run metadata for', i)

            # Save the variables to disk
            save_path = saver.save(sess, ckpt_dir + ckpt_file)
            print("Model saved in file: %s" % save_path)

        # Train step
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    save_path = saver.save(sess, ckpt_dir + ckpt_file)
    print("Model saved in file: %s" % save_path)

    writer.close()

    # Get data reader
    data_reader = DataReader(
        "D:\\Stuff\\Faks\\BIOINF\\Projekt\\localization_data\\color_testing\\",
        batch_size=2,
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
    1500,
    "D:\\Stuff\\Faks\\BIOINF\\Projekt\\fin_or_not\\",
    "model.ckpt"
)