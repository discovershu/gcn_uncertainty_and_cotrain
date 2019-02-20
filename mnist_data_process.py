# from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import scipy.sparse as sp
import networkx as nx

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_mnist_data(seed):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    first_image = mnist.test.images[0]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))

    a = 0  # 0 116
    b = 0  # 0.5 71
    c = 0  # 1.0 1
    d = 0  # 0.9 39
    e = 0  # 0.6 66
    f = 0  # 0.7 62
    g = 0  # 0.8 50

    first_image_new = []
    for i in range(len(first_image)):
        if first_image[i] > 0.8:
            first_image_new.append(1.0)
        else:
            first_image_new.append(0.0)
    first_image_new = np.array(first_image_new, dtype='float')
    pixels_new = first_image_new.reshape((28, 28))

    # show fig
    # plt.imshow(pixels_new)
    # plt.imshow(pixels, cmap='gray')
    # plt.show()

    # get features
    features = first_image.reshape((len(first_image), 1))
    features = sp.csr_matrix(features)

    # get node label
    labels = np.zeros([len(first_image), 2], dtype=np.float)
    for i in range(len(first_image)):
        if first_image[i] > 0.8:
            labels[i][1] = 1.
        else:
            labels[i][0] = 1.

    # get adj
    graph = dict()
    m = 0
    temp = []
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            left = m - 1
            right = m + 1
            up = m - pixels.shape[0]
            down = m + pixels.shape[0]
            p = int(m / pixels.shape[0])
            left_t = p * pixels.shape[0]
            right_t = (p + 1) * pixels.shape[0]
            if left_t <= left < right_t:
                temp.append(m - 1)
                if ((left_t - pixels.shape[0]) <= (left - pixels.shape[0]) < (right_t - pixels.shape[0])) and (
                        0 <= (left_t - pixels.shape[0])):
                    temp.append(m - 1 - pixels.shape[0])
                if ((left_t + pixels.shape[0]) <= (left + pixels.shape[0]) < (right_t + pixels.shape[0])) and (
                        (right_t + pixels.shape[0]) <= (pixels.shape[0] * pixels.shape[1])):
                    temp.append(m - 1 + pixels.shape[0])
            if left_t <= right < right_t:
                temp.append(m + 1)
                if ((left_t - pixels.shape[0]) <= (right - pixels.shape[0]) < (right_t - pixels.shape[0])) and (
                        0 <= (left_t - pixels.shape[0])):
                    temp.append(m + 1 - pixels.shape[0])
                if ((left_t + pixels.shape[0]) <= (right + pixels.shape[0]) < (right_t + pixels.shape[0])) and (
                        (right_t + pixels.shape[0]) <= (pixels.shape[0] * pixels.shape[1])):
                    temp.append(m + 1 + pixels.shape[0])
            if 0 <= up < (pixels.shape[0] * pixels.shape[1]):
                temp.append(up)
            if 0 <= down < (pixels.shape[0] * pixels.shape[1]):
                temp.append(down)

            graph[m] = temp
            m = m + 1
            temp = []

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # get train/val/test mask
    remaining_indices = list(range(labels.shape[0]))
    train_dict = dict()
    train_dict_temp = []
    train_indices = []

    for j in range(labels.shape[1]):
        for i in range(labels.shape[0]):
            if labels[i][j] == 1.0:
                if len(train_dict_temp) < 300:
                    train_dict_temp.append(i)
        train_dict[j] = train_dict_temp
        train_dict_temp = []

    for i in range(len(train_dict)):
        for j in range(len(train_dict[i])):
            train_indices.append(train_dict[i][j])

    train_mask = sample_mask(train_indices, labels.shape[0])
    val_mask = sample_mask(remaining_indices, labels.shape[0])
    test_mask = sample_mask(remaining_indices, labels.shape[0])

    # get y_train/y_val/y_test
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask