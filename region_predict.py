# -*- coding: utf-8 -*-
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

n_neurons = 100
n_steps = 40
n_inputs = 1
n_outputs = 1
learning_rate = 0.001
n_iterations = 500
batch_size = 50
n_train_split = 0.75
fname_model = './region_predict_model'


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def show_ts(lst):
    plt.plot(lst)
    plt.show()


def get_bw():
    bw_old = 0
    fname = "/Users/wangzhiguo/Downloads/k2_201803.csv"
    lst = []
    with open(fname) as f:
        for line in f:
            if line.startswith('ZheJiang_CT,'):
                _, t, bw = line.strip().split(',')
                if not bw:
                    bw = bw_old
                else:
                    bw = float(bw)
                    bw_old = bw
                lst.append(bw)
    # print(len(lst))
    return np.array(lst)

# TODO global
K2_ZJ_ts = get_bw().reshape((-1, 1))
N_train = int(len(K2_ZJ_ts) * n_train_split - n_steps)

def time_series(a):
    return K2_ZJ_ts[a]


def next_batch(batch_size, n_steps, start_index):
    """generate random batch_size
    return: X_batch, Y_batch
    """
    t0 = np.random.randint(start_index, size=(batch_size, 1))
    Ts = t0 + np.arange(0, n_steps + 1)
    ys = time_series(Ts)
    # one step after
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(
        -1, n_steps, 1)


def show_predict(t_instance, y_pred):
    plt.title("Testing the model", fontsize=14)
    plt.plot(
        t_instance[:-1],
        time_series(t_instance[:-1]),
        "bo",
        markersize=10,
        label="instance")
    plt.plot(
        t_instance[1:],
        time_series(t_instance[1:]),
        "w*",
        markersize=10,
        label="target")
    plt.plot(
        t_instance[1:],
        y_pred[0, :, 0],
        "r.",
        markersize=10,
        label="prediction")
    plt.legend(loc="upper left")
    plt.xlabel("Time")

    plt.show()


def main(training=True):
    reset_graph()

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if training:
            init.run()
            for iteration in range(n_iterations):
                X_batch, y_batch = next_batch(batch_size, n_steps, N_train)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                if iteration % 100 == 0:
                    mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                    print(iteration, "\tMSE:", mse)
            saver.save(sess, fname_model)
        else:
            saver.restore(sess, fname_model)
            t_instance = np.array(range(N_train + 1, N_train + 2 + n_steps))
            X_new = time_series(np.array(t_instance[:-1].reshape(1, -1)))
            y_pred = sess.run(outputs, feed_dict={X: X_new})
            show_predict(t_instance, y_pred)


if __name__ == "__main__":
    # main()
    main(False)
