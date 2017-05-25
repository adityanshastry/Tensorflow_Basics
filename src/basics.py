import numpy as np
import tensorflow as tf


def constant_tensors():

    # no inputs taken, and stores a value internally

    # declare the constant variables
    node_1 = tf.constant(value=17.3, dtype=tf.float32)
    node_2 = tf.constant(value=21.5, dtype=tf.float32)

    # create a session
    c_sess = tf.Session()

    # print tensor value
    print 'Node 1: ', c_sess.run(node_1)
    print 'Node 2: ', c_sess.run(node_2)


def constant_tensor_arithmetic():

    # declare and initialize the constant tensors
    node_1 = tf.constant(value=17.3, dtype=tf.float32)
    node_2 = tf.constant(value=21.5, dtype=tf.float32)

    # create a session
    a_sess = tf.Session()

    # print tensor value
    print 'Node 1: ', a_sess.run(node_1)
    print 'Node 2: ', a_sess.run(node_2)

    # tensor arithmetic operations
    print 'Addition: ', a_sess.run(node_1 + node_2)
    print 'Subtraction: ', a_sess.run(node_1 - node_2)
    print 'Multiplication: ', a_sess.run(node_1 * node_2)
    print 'Division:', a_sess.run(node_1 / node_2)


def tensor_placeholders():

    # no value initialized while declared, the values are provided at run time
    # The shape can vary based on the input provided

    # declare the placeholder variables
    node_1 = tf.placeholder(dtype=tf.float32)
    node_2 = tf.placeholder(dtype=tf.float32)

    # create a session
    p_sess = tf.Session()

    # assign values to the placeholder and print
    print 'Node 1 placeholder: ', p_sess.run(fetches=node_1, feed_dict={node_1: 17.3})
    print 'Node 2 placeholder: ', p_sess.run(fetches=node_2, feed_dict={node_2: [23.5, 19.23]})


def tensor_variables():

    # Trainable parameters for the network/model

    # declare the variables
    var_1 = tf.Variable(initial_value=[1, 2], dtype=tf.float32)
    var_2 = tf.Variable(initial_value=[[3, 4], [5, 10]], dtype=tf.float32)

    # declare the placeholder
    x = tf.placeholder(dtype=tf.float32)

    # create a session
    v_sess = tf.Session()

    # initialize the variables
    v_sess.run(tf.global_variables_initializer())

    # a sample linear model
    print v_sess.run(fetches=tf.matmul(x, var_2) + var_1, feed_dict={x: [[1, 2], [3, 4]]})


def tensor_train():

    # tensorflow train API

    # declare the variables
    var_1 = tf.Variable(initial_value=[1, 2], dtype=tf.float32)
    var_2 = tf.Variable(initial_value=[[3, 4], [5, 10]], dtype=tf.float32)

    # declare placeholders
    x = tf.placeholder(dtype=tf.float32)
    y = tf.placeholder(dtype=tf.float32)

    # create a session
    t_sess = tf.Session()

    # declare a loss function, and an optimizer for the loss function
    model_loss = tf.reduce_sum(input_tensor=tf.square(x=tf.matmul(x, var_2) + var_1 - y))
    model_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(model_loss)

    # initialize the variables
    t_sess.run(tf.global_variables_initializer())

    # model training
    for i in xrange(1000):
        t_sess.run(fetches=model_optimizer, feed_dict={x: [[1, 2], [3, 4]], y: [12, 13]})

    # final loss
    print t_sess.run(fetches=model_loss, feed_dict={x: [[1, 2], [3, 4]], y: [12, 13]})


def main():
    # constant_tensors()
    # constant_tensor_arithmetic()
    # tensor_placeholders()
    # tensor_variables()
    tensor_train()


if __name__ == '__main__':
    main()
