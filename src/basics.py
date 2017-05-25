import numpy as np
import tensorflow as tf


def constant_tensors():

    # no inputs taken, and stores a value internally
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




def main():
    # constant_tensors()
    constant_tensor_arithmetic()


if __name__ == '__main__':
    main()
