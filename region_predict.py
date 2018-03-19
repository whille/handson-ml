# -*- coding: utf-8 -*-
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

n_inputs = 2
n_outputs = 1
n_neurons = 100
n_steps = 80    # >
learning_rate = 0.001
n_iterations = 1000 # >
batch_size = 50
n_train_split = 0.75
fname_model = './region_predict_model'
n_len = 1440


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


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
                lst.append((bw, int(t)%(3600*24)/3600))   # hour
    return np.array(lst)

# TODO global
K2_ZJ_ts = get_bw()
N_train = int(len(K2_ZJ_ts) * n_train_split - n_steps)

def time_series(ts):
    return K2_ZJ_ts[ts]


def next_batch(n_steps, end, n_future):
    """generate random batch_size
    return: X_batch, y_batch
    """
    t0 = np.random.randint(end, size=(batch_size, 1))
    Ts = t0 + np.arange(0, n_steps + n_future)
    ys = time_series(Ts)
    # one step after
    return ys[:, :-n_future, :], ys[:, n_future:, 0].reshape(-1, n_steps, n_outputs)


def show_predict(t_instance, Ys, lst_t):
    plt.title("Testing the model", fontsize=14)
    plt.plot(
        t_instance[:n_steps],
        time_series(t_instance[:n_steps])[:, 0],
        "bo-",
        label="instance")
    plt.plot(
        t_instance[n_steps:],
        time_series(t_instance[n_steps:])[:, 0],
        "g--",
        label="target")
    for i, n_future in enumerate(lst_t):
        plt.plot(
            t_instance[n_steps + n_future:n_steps + n_future + n_len],
            Ys[i],
            label="predict_%s" %n_future)
    plt.legend(loc="upper left")
    plt.xlabel("Time")
    plt.show()


def main(n_future, training=True, t_instance = None):
    reset_graph()

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

    loss = tf.reduce_mean(tf.divide(tf.abs(outputs - y), y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    fname = '%s_%s' %(fname_model, n_future)

    with tf.Session() as sess:
        if training:
            init.run()
            for iteration in range(n_iterations):
                X_batch, y_batch = next_batch(n_steps, N_train, n_future)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                if iteration % 100 == 0:
                    mre = loss.eval(feed_dict={X: X_batch, y: y_batch})
                    # mean relative error
                    print(iteration, "\tmre: %.1f%%" %(100. * mre))
            saver.save(sess, fname)
        else:
            saver.restore(sess, fname)
            Y_pred = []
            for i in range(n_len):
                X_batch = time_series(t_instance[i:i+n_steps]).reshape(1, n_steps, -1)
                y_pred = sess.run(outputs, feed_dict={X: X_batch})
                Y_pred.append(y_pred[0, -1, 0])
            return Y_pred

if __name__ == "__main__":
    # import sys
    # n_future = 5 # to predict next minutes
    # if len(sys.argv) > 1:
        # n_future = int(sys.argv[1])
    lst_t = range(25, 31, 5)
    Ys = []
    n_start = np.random.randint(N_train, len(K2_ZJ_ts) - n_len - lst_t[-1] - n_steps)
    t_instance = np.array(range(n_len + lst_t[-1] + n_steps)) + n_start
    for n_future in lst_t:
        print("n_future", n_future)
        main(n_future)
        Ys.append(main(n_future, False, t_instance))
    show_predict(t_instance, Ys, lst_t)
