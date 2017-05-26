from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os


def get_weight_variable(shape):
    """
    The weight is initialized with values drawn from a truncated normal distribution of mean = 0, and stddev = 0.1
    :param shape: shape of the weight
    :return: Variable object for the weight
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def get_bias_variable(shape):
    """
    Inintializes a weigt parameter for bias, a constant value is used
    :param shape: shape of the weight
    :return: Variable object for the weight
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_2d(x, w):
    """
    Perform convolutions on an image, with the given filter, with strides 1
    The output feature maps are the same size as the input image due to 'SAME' param to conv2d
    :param x: input image
    :param w: filter for convolution
    :return: feature maps after applying the filter on the image
    """
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, pool_size):
    """
    Perform max pooling on the provided feature maps.
    The output will be of the size (input_width/pool_size[0], input_height/pool_size[1])
    :param x: input feature map from a convolution layer
    :param pool_size: the size of the pool coverage
    :return: feature maps after performing size=pool_size max pooling
    """
    return tf.nn.max_pool(x, ksize=[1, pool_size[0], pool_size[1], 1], strides=[1, pool_size[0], pool_size[1], 1],
                          padding='SAME')


def conv_layers_for_mnist(mnist_data):
    """
    MNIST classification using deep convolution (, and pooling) layers, and a dense feedforward network
    """

    # With InteractiveSession, can modify the computation graph after creation of session
    sess = tf.InteractiveSession()

    # Placeholder tensors for input images, and output labels
    # size of image - 28X28, number of classes - 10
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # weights for conv layer 1
    # format of weight - [(filter size), input_channels, output_channels]
    w_conv_1 = get_weight_variable(shape=[5, 5, 1, 32])
    b_conv_1 = get_bias_variable([32])

    # weights for conv layer 2
    w_conv_2 = get_weight_variable(shape=[5, 5, 32, 64])
    b_conv_2 = get_bias_variable([64])

    # weights for dense layer 1
    w_dense_1 = get_weight_variable(shape=[7 * 7 * 64, 500])
    b_dense_1 = get_bias_variable(shape=[500])

    # weights for dense output layer
    w_dense_output = get_weight_variable(shape=[500, 10])
    b_dense_output = get_bias_variable(shape=[10])

    # reshape image according to conv layer requirements - [-1, width, height, channels]
    x_reshape = tf.reshape(tensor=x, shape=[-1, 28, 28, 1])

    # conv layer 1
    h_conv_1 = tf.nn.relu(conv_2d(x=x_reshape, w=w_conv_1) + b_conv_1)
    h_pool_1 = max_pool(x=h_conv_1, pool_size=(2, 2))

    # conv layer 2
    h_conv_2 = tf.nn.relu(conv_2d(x=h_pool_1, w=w_conv_2) + b_conv_2)
    h_pool_2 = max_pool(x=h_conv_2, pool_size=(2, 2))

    # reshape the max pool output according to dense layer requirements - [-1, width * height * channels]
    x_flattened = tf.reshape(tensor=h_pool_2, shape=[-1, 7 * 7 * 64])

    # dense layer 1
    d_dense_1 = tf.nn.relu(tf.matmul(a=x_flattened, b=w_dense_1) + b_dense_1)

    # dropout for regularization
    keep_rate = tf.placeholder(dtype=tf.float32)
    h_dense_dropout = tf.nn.dropout(x=d_dense_1, keep_prob=keep_rate)

    # dense output layer
    y = tf.matmul(a=h_dense_dropout, b=w_dense_output) + b_dense_output

    # loss
    model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # optimizer
    model_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss=model_loss)

    # prediction, and accuracy metric
    accuracy = tf.reduce_mean(
        input_tensor=tf.cast(x=tf.equal(x=tf.argmax(input=y, axis=1), y=tf.argmax(input=y_, axis=1)), dtype=tf.float32))
    tf.summary.scalar(name="Accuracy", tensor=accuracy)

    # Summary tensor
    mnist_summary = tf.summary.merge_all()

    # saver for writing checkpoints
    mnist_checkpoints = tf.train.Saver()

    # summary writer to output summaries and the graph
    mnist_summary_writer = tf.summary.FileWriter(logdir="../data/summary", graph=sess.graph)

    # initialize variables
    sess.run(tf.global_variables_initializer())

    # training
    for i in xrange(10000):
        data, labels = mnist_data.train.next_batch(50)
        feed_dict = {x: data, y_: labels, keep_rate: 0.7}
        model_optimizer.run(feed_dict=feed_dict)

        if i % 100 == 0:
            print 'Training Step: {}, Accuracy: {}'.format(i + 1, accuracy.eval(feed_dict=feed_dict))
            train_summary = sess.run(fetches=mnist_summary, feed_dict=feed_dict)
            mnist_summary_writer.add_summary(summary=train_summary, global_step=i)
            mnist_summary_writer.flush()

        if (i+1) % 1000 == 0:
            print 'Checkpointing at step ', i+1
            checkpoint_file = os.path.join("../data/checkpoints", "mnist_model.checkpoint")
            mnist_checkpoints.save(sess=sess, save_path=checkpoint_file, global_step=i)

    # Testing
    print 'Test Accuracy: ', accuracy.eval(
        feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels, keep_rate: 1.0})


def main():
    mnist_data = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
    conv_layers_for_mnist(mnist_data=mnist_data)


if __name__ == '__main__':
    main()
