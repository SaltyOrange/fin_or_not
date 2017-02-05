from utils import DataReader
from classification_net.net import inference

import tensorflow as tf

import sys


train_dataset_dir = sys.argv[1]
test_dataset_dir = sys.argv[2]
iterations = sys.argv[3]
batch_size = sys.argv[4]
logfile = sys.argv[5]

ckpt_file = "ckpt"

x, y_true, y, gap_w, conv3_pool, train_step, accuracy = inference()

sess = tf.InteractiveSession()

# Setup summary
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logfile, sess.graph)

# Get data reader
data_reader = DataReader(train_dataset_dir, batch_size=batch_size,
                         file_names=False)

# Add ops to save and restore all the variables
saver = tf.train.Saver()

tf.add_to_collection('x', x)
tf.add_to_collection('y', y)

tf.add_to_collection('gap_w', gap_w)
tf.add_to_collection('conv3', conv3_pool)

ckpt = tf.train.latest_checkpoint(".")
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
            feed_dict={x: batch[0], y_true: batch[1]}
        )
        print("Training step %d, training accuracy %g" %
              (i, train_accuracy))

        # Calculate and print Cumulative moving average
        n += 1
        cma += (train_accuracy - cma) / n
        print("Cumulative moving average: %f" % cma)

        # Generate summary
        summary = sess.run(
            merged, feed_dict={x: batch[0], y_true: batch[1]}
        )

        run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        writer.add_run_metadata(run_metadata, 'step%05d' % i)
        writer.add_summary(summary, i)
        print('Adding run metadata for', i)

    if i % 100 == 0:
        # Save the variables to disk
        save_path = saver.save(sess, "./" + ckpt_file)
        print("Model saved in file: %s" % save_path)

    # Train step
    train_step.run(feed_dict={x: batch[0], y_true: batch[1]})

save_path = saver.save(sess, "./" + ckpt_file)
print("Model saved in file: %s" % save_path)

# Get data reader
data_reader = DataReader(test_dataset_dir, batch_size=1, file_names=True)

for i in range(30):
    # Get next batch of images and labels
    batch = data_reader.next()

    prediction = sess.run(y, feed_dict={x: batch[0], y_true: batch[1]})
    prediction = tf.argmax(prediction, 1).eval()
    print("Prediction for %s is %s (0 - fin, 1 - no fin):"
          % (batch[2], prediction))

    # Get and print accuracy
    test_accuracy = accuracy.eval(
        feed_dict={x: batch[0], y_true: batch[1]}
    )
    print("Testing step %d, testing accuracy %g" % (i, test_accuracy))
