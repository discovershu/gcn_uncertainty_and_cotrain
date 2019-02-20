from PIL import Image
import numpy as np
import networkx as nx
import scipy.sparse as sp

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_voc_data_multiclass(seed, test_ratio, img_name, scribble):
    # read images
    img_ori = Image.open('/network/rit/lab/ceashpc/shuhu/GCN/data/{}.jpg'.format(img_name))  # 960, 720 2007_002852
    img_ground_truth = Image.open('/network/rit/lab/ceashpc/shuhu/GCN/data/{}.png'.format(img_name))  # 960, 720 2007_002852
    if scribble == True:
        img_ori_scribble = Image.open('/network/rit/lab/ceashpc/shuhu/GCN/data/{}_scribble.jpg'.format(img_name))

    # crop images
    # img_ori_crop = img_ori.crop((0, 0, 500, 320))
    # img_ground_truth_crop = img_ground_truth.crop((0, 0, 500, 320))
    # img_ori_crop.show()

    # resize images
    # new_img_ori = img_ori_crop.resize((250, 160))
    # new_img_ground_truth = img_ground_truth_crop.resize((250, 160))
    # new_img_ori.show()
    # new_img_ground_truth.show()

    # get RGB
    # RGB_ori = np.array(new_img_ori)
    # RGB_ground_truth = np.array(new_img_ground_truth)

    RGB_ori = np.array(img_ori)
    RGB_ground_truth = np.array(img_ground_truth)
    if scribble == True:
        RGB_ori_scribble = np.array(img_ori_scribble)
        red = np.asarray([230, 50, 50])
        blue = np.asarray([50, 50, 230])

    # normalize
    RGB_ori_norm = np.true_divide(RGB_ori, 255)
    RGB_ground_truth_norm = np.true_divide(RGB_ground_truth, 255)

    # get node label

    labels = np.zeros([RGB_ground_truth.shape[0] * RGB_ground_truth.shape[1], 4], dtype=np.float)
    labels_index = 0
    train_index_scribble = []
    for i in range(RGB_ground_truth.shape[0]):
        for j in range(RGB_ground_truth.shape[1]):
            if RGB_ground_truth[i][j] == 0 or RGB_ground_truth[i][j] == 255:
                labels[labels_index][0] = 1.
            if RGB_ground_truth[i][j] == 11:
                labels[labels_index][1] = 1.
            if RGB_ground_truth[i][j] == 15:
                labels[labels_index][2] = 1.
            if RGB_ground_truth[i][j] == 9:
                labels[labels_index][3] = 1.
            if scribble == True:
                if RGB_ori_scribble[i][j][0] > red[0] and RGB_ori_scribble[i][j][1] < red[1] and RGB_ori_scribble[i][j][2] < red[2]:
                    train_index_scribble.append(labels_index)
                if RGB_ori_scribble[i][j][0] < blue[0] and RGB_ori_scribble[i][j][1] < blue[1] and RGB_ori_scribble[i][j][2] > blue[2]:
                    train_index_scribble.append(labels_index)
            labels_index = labels_index + 1
    # np.save("C:\\Users\\Shu\\Desktop\\image_labels_for_gcn_240_180.npy", labels)

    print(np.sum(labels, axis=0))

    # get features
    features = []
    for i in range(RGB_ori_norm.shape[0]):
        for j in range(RGB_ori_norm.shape[1]):
            features.append(RGB_ori_norm[i][j])
    # np.save("C:\\Users\\Shu\\Desktop\\image_features_for_gcn_240_180.npy", features)
    features = sp.csr_matrix(features)

    # get adj
    graph = dict()
    m = 0
    temp = []
    for i in range(RGB_ori_norm.shape[0]):
        for j in range(RGB_ori_norm.shape[1]):
            left = m - 1
            right = m + 1
            up = m - RGB_ori_norm.shape[0]
            down = m + RGB_ori_norm.shape[0]
            p = int(m / RGB_ori_norm.shape[0])
            left_t = p * RGB_ori_norm.shape[0]
            right_t = (p + 1) * RGB_ori_norm.shape[0]
            if left_t <= left < right_t:
                temp.append(m - 1)
                if ((left_t - RGB_ori_norm.shape[0]) <= (left - RGB_ori_norm.shape[0]) < (
                        right_t - RGB_ori_norm.shape[0])) and (0 <= (left_t - RGB_ori_norm.shape[0])):
                    temp.append(m - 1 - RGB_ori_norm.shape[0])
                if ((left_t + RGB_ori_norm.shape[0]) <= (left + RGB_ori_norm.shape[0]) < (
                        right_t + RGB_ori_norm.shape[0])) and (
                        (right_t + RGB_ori_norm.shape[0]) <= (RGB_ori_norm.shape[0] * RGB_ori_norm.shape[1])):
                    temp.append(m - 1 + RGB_ori_norm.shape[0])
            if left_t <= right < right_t:
                temp.append(m + 1)
                if ((left_t - RGB_ori_norm.shape[0]) <= (right - RGB_ori_norm.shape[0]) < (
                        right_t - RGB_ori_norm.shape[0])) and (0 <= (left_t - RGB_ori_norm.shape[0])):
                    temp.append(m + 1 - RGB_ori_norm.shape[0])
                if ((left_t + RGB_ori_norm.shape[0]) <= (right + RGB_ori_norm.shape[0]) < (
                        right_t + RGB_ori_norm.shape[0])) and (
                        (right_t + RGB_ori_norm.shape[0]) <= (RGB_ori_norm.shape[0] * RGB_ori_norm.shape[1])):
                    temp.append(m + 1 + RGB_ori_norm.shape[0])
            if 0 <= up < (RGB_ori_norm.shape[0] * RGB_ori_norm.shape[1]):
                temp.append(up)
            if 0 <= down < (RGB_ori_norm.shape[0] * RGB_ori_norm.shape[1]):
                temp.append(down)

            graph[m] = temp
            m = m + 1
            temp = []

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # get train/val/test mask
    # seed = 123
    # test_size = None

    remaining_indices = list(range(labels.shape[0]))
    train_dict = dict()
    train_dict_temp = []
    train_indices = []
    train_0 = []
    train_1 = []
    random_state = np.random.RandomState(seed)

    for j in range(labels.shape[1]):
        for i in range(labels.shape[0]):
            if labels[i][j] == 1.0:
                if j == 0:
                    train_0.append(i)
                else:
                    train_1.append(i)
    train_indices_0 = random_state.choice(train_0, int(0.1 * len(train_0)), replace=False)
    train_indices_1 = random_state.choice(train_1, int(1.0 * len(train_1)), replace=False)
    for i in range(len(train_indices_0)):
        train_indices.append(train_indices_0[i])
    for i in range(len(train_indices_1)):
        train_indices.append(train_indices_1[i])

    if scribble == True:
        train_mask = sample_mask(train_index_scribble, labels.shape[0])
    else:
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