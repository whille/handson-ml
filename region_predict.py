# -*- coding: utf-8 -*-
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import re
import codecs
import os

N_train = 1e4
n_inputs = 4
n_outputs = 1
n_neurons = 200  # >
n_steps = 100  # <
n_test = 1440
learning_rate = 1e-3  # <
n_iterations = 5000  # >
batch_size = 50
ERR_DELTA = 0.3
fname_model = './region_predict_model'

home = os.path.expanduser("~")

# TODO penalize less bw


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def init_region():
    f_region = os.sep.join((home, 'Documents/region.csv'))
    f = codecs.open(f_region, encoding='utf-8')
    headers = f.readline().strip().split(',')
    i_lon, i_lat, i_enname = [
        headers.index(i) for i in ('lon', 'lat', 'enName')
    ]
    dic = {}
    for l in f:
        lst = l.strip().split(',')
        lon, lat = lst[i_lon], lst[i_lat]
        if not lon or not lat:
            continue
        lon = int(float(lon))
        lat = int(float(lat))
        if lon < 0 or lat < 0:
            continue
        dic[lst[i_enname]] = (lon, lat)
    return dic


def get_bw():
    dic_region = init_region()
    # TODO k2_last30_s.csv
    fname = os.sep.join((home, 'Downloads/k2_201803_sort.csv'))
    lst = []
    region_old, bw_old = '', 0
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            region, t, bw = line.split(',')
            if not bw:
                continue
            # TODO other OP
            if region not in dic_region or not region.endswith('_CT'):
                continue
            bw = float(bw)
            ratio = bw / max(bw_old, 1e-3)
            if region == region_old and (ratio > 1 / ERR_DELTA
                                         or ratio < ERR_DELTA):
                print(line)
            if region != region_old:
                lst.append([])
            # bw, 10min, longitude, latitude
            lst[-1].append((bw, int(t) % (3600 * 24) / 600,
                            dic_region[region][0], dic_region[region][1]))
            region_old, bw_old = region, bw
    res = []
    for sub_lst in lst:
        if len(sub_lst) < N_train:
            continue
        res.append(np.array(sub_lst))
    return res


# TODO global
K2_bw_ts = get_bw()
N_data = len(K2_bw_ts)


def time_series(ns, ts):
    lst = []
    for i in range(len(ns)):
        lst.append(K2_bw_ts[ns[i]][ts[i]])
    return np.stack(lst)


def next_batch(n_future, start=0):
    """generate random batch_size
    return: X_batch, y_batch
    """
    Ns = np.random.randint(N_data, size=batch_size)
    Ts = np.random.randint(
        start, N_train, size=(batch_size, 1)) + np.arange(
            0, n_steps + n_future)
    ys = time_series(Ns, Ts)
    # one step after
    return ys[:, :-n_future, :], ys[:, n_future:, 0].reshape(
        -1, n_steps, n_outputs)


def show_predict(n, test_ts, Ys, lst_t):
    # plt.title("Testing the model", fontsize=14)
    plt.plot(
        test_ts[:n_steps],
        time_series(n, test_ts[:n_steps].reshape(1, -1))[0, :, 0],
        "bo-",
        label="instance")
    plt.plot(
        test_ts[n_steps:],
        time_series(n, test_ts[n_steps:].reshape(1, -1))[0, :, 0],
        "g--",
        label="target")
    for i, n_future in enumerate(lst_t):
        plt.plot(test_ts[n_steps + n_future:n_steps + n_future + n_test], Ys[i], label="predict_%s" % n_future)
        X_, Y_poly = polyfit(n, n_start, n_future)
        plt.plot(X_, Y_poly, label="polyfit_%s" % n_future)
    plt.legend(loc="best")
    plt.xlabel("Time")
    # plt.savefig('t.png')
    plt.show()


def main(n_future, n=None, test_ts=None, run_on=False):
    reset_graph()
    training = (n is None)

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    cell = tf.contrib.rnn.BasicRNNCell(
        num_units=n_neurons, activation=tf.nn.relu)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    # loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, outputs)))))
    loss = tf.reduce_mean(tf.divide(tf.abs(outputs - y), y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    saver = tf.train.Saver()
    fname = '%s_%s' % (fname_model, n_future)

    with tf.Session() as sess:
        if training:
            if run_on:
                saver.restore(sess, fname)
            else:
                init = tf.global_variables_initializer()
                init.run()
            for iteration in range(n_iterations):
                X_batch, y_batch = next_batch(n_future)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                if iteration % 100 == 0:
                    mre = loss.eval(feed_dict={X: X_batch, y: y_batch})
                    # mean relative error
                    print(iteration, "\tmre: %.1f%%" % (100. * mre))
            saver.save(sess, fname)
            return mre
        else:
            saver.restore(sess, fname)
            Y_pred = []
            for i in range(n_test):
                X_test = time_series(n, test_ts[i:i + n_steps].reshape(1, -1))
                y_pred = sess.run(outputs, feed_dict={X: X_test})
                Y_pred.append(y_pred[0, -1, 0])
            return Y_pred


def modify_test(n_start, n_test):
    dic_modify = {(0, 0.2): 0, (0.4, 0.6): 2, (0.8, 1.0): 0}
    for (start, end), ratio in dic_modify.iteritems():
        print(n_start + int(start * n_test), n_start + int(end * n_test))
        K2_bw_ts[n[0]][n_start + int(start * n_test):
                       n_start + int(end * n_test), 0] *= ratio


def polyfit(n, n_start, n_future):
    Y = K2_bw_ts[n[0]][n_start:n_start + n_test, 0]
    Y_pred = []
    for start in range(n_steps, len(Y) - n_steps):
        y = Y[start - n_steps:start + 1]
        x = np.arange(len(y)) + start
        z1 = np.polyfit(x, y, 2)  # 自由度为2
        p1 = np.poly1d(z1)
        Y_pred.append(p1(x[-1] + n_future))
    Y_pred = np.array(Y_pred)
    X_ = np.arange(n_steps, n_steps + len(Y_pred)) + n_start
    return X_, Y_pred
    plt.plot(np.arange(len(Y)) + n_start, Y, label='Origin Line')
    plt.plot(X_, Y_pred)


if __name__ == "__main__":
    import sys
    run_on = False
    if len(sys.argv) > 1:
        run_on = sys.argv[1] == 'True'
    lst_t = range(20, 31, 10)
    Ys = []
    n = np.random.randint(N_data, size=1)
    n_start = np.random.randint(N_train, N_train + 1000)
    test_ts = np.array(range(n_test + lst_t[-1] + n_steps)) + n_start
    modify_test(n_start, n_test)
    dic_mre = {}
    for n_future in lst_t:
        print("n_future:", n_future)
        # dic_mre[n_future] = main(n_future, run_on=run_on)
        Ys.append(main(n_future, n, test_ts, run_on=run_on))
    print("summary:")
    print('random n:', n[0], 'start_t:', n_start)
    for i, mre in dic_mre.iteritems():
        print(i, "%.1f%%" % (mre * 100))
    show_predict(n, test_ts, Ys, lst_t)
