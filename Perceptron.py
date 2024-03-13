# single node perceptron

import matplotlib.pyplot as plt
import random


def compute_output(w, x):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i]
    if z < 0:
        return -1
    else:
        return 1


def perceptron_training(x_train, y_train, w, LEARNING_RATE):
    index_list = [i for i in range(len(x_train))]
    all_correct = False
    while not all_correct:
        all_correct = True
        random.shuffle(index_list)
        for i in index_list:
            x = x_train[i]
            y = y_train[i]
            p_out = compute_output(w, x)
            if y != p_out:
                for j in range(0, len(w)):
                    w[j] += (y * LEARNING_RATE * x[j])
                all_correct = False
                show_learning(w)


def show_learning(w):
    global color_index
    print('w0=', '%5.2f' % w[0], ' , w1 =', '%5.2f' % w[1], ' , w2 =', '%5.2f' % w[2])
    if color_index == 0:
        plt.plot([1.0], [1.0], 'b_', markersize=12)
        plt.plot([-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], 'r+', markersize=12)
        plt.axis([-2, 2, -2, 2])
        plt.xlabel('x1')
        plt.ylabel('x2')
    x = [-2.0, 2.0]
    if abs(w[2]) < 1e-5:
        y = [-w[1] / 1e-5 * (-2.0) + (w[0] / 1e-5), -w[1] / 1e-5 * 2.0 + (w[0] / 1e-5)]
    else:
        y = [-w[1] / w[2] * (-2.0) + (-w[0] / w[2]), -w[1] / w[2] * 2.0 + (-w[0] / w[2])]
    plt.plot(x, y, color_list[color_index])
    if color_index < (len(color_list) - 1):
        color_index += 1


color_list = ['r-', 'm-', 'y-', 'c-', 'b-', 'g-']
color_index = 0

LEARNING_RATE = 1

x_train = [(1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0)]
y_train = [1.0, 1.0, 1.0, -1.0]

w = [0, 0, 0]

perceptron_training(x_train, y_train, w, LEARNING_RATE)
