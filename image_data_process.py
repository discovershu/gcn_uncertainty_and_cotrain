from PIL import Image
import numpy as np
import networkx as nx

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_image_data(seed):
    #read images
    img_ori = Image.open('C:\\Users\\Shu\\Desktop\\CamSeq01\\0016E5_07959.png') #960, 720
    img_ground_truth = Image.open('C:\\Users\\Shu\\Desktop\\CamSeq01\\0016E5_07959_L.png') #960, 720

    #resize images
    new_img_ori = img_ori.resize((240,180))
    new_img_ground_truth = img_ground_truth.resize((240,180))

    #get RGB
    RGB_ori = np.array(new_img_ori)
    RGB_ground_truth = np.array(new_img_ground_truth)

    #normalize
    RGB_ori_norm = np.true_divide(RGB_ori, 255)
    RGB_ground_truth_norm = np.true_divide(RGB_ground_truth, 255)

    # read class label
    total_class_label = []

    with open("C:\\Users\\Shu\\Desktop\\CamSeq01\\label_colors.txt", "r") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        class_label = []
        class_label.append(int(lines[i].split()[0]))
        class_label.append(int(lines[i].split()[1]))
        class_label.append(int(lines[i].split()[2]))
        total_class_label.append(class_label)

    # generate node label
    # node_label = []
    # for i in range(RGB_ground_truth.shape[0]):
    #     for j in range(RGB_ground_truth.shape[1]):
    #         for m in range(len(total_class_label)):
    #             if (RGB_ground_truth[i][j][0]==total_class_label[m][0]) and (RGB_ground_truth[i][j][1]==total_class_label[m][1]) and (RGB_ground_truth[i][j][2]==total_class_label[m][2]):
    #                 node_label.append(m)
    # np.save("C:\\Users\\Shu\\Desktop\\image_node_label_240_180.npy", node_label)

    # get node label
    node_label = np.load("C:\\Users\\Shu\\Desktop\\image_node_label_240_180.npy")
    node_label_unique = np.unique(node_label)
    node_label_unique_dict = dict()
    for i in range(len(node_label_unique)):
        node_label_unique_dict[node_label_unique[i]] = i

    labels = np.zeros([len(node_label), len(node_label_unique)], dtype=np.float)
    for i in range(len(node_label)):
        labels[i][node_label_unique_dict[node_label[i]]] = 1.
    # np.save("C:\\Users\\Shu\\Desktop\\image_labels_for_gcn_240_180.npy", labels)

    # get features
    features = []
    for i in range(RGB_ori_norm.shape[0]):
        for j in range(RGB_ori_norm.shape[1]):
            features.append(RGB_ori_norm[i][j])
    # np.save("C:\\Users\\Shu\\Desktop\\image_features_for_gcn_240_180.npy", features)

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
            right_t = (p+1) * RGB_ori_norm.shape[0]
            if left_t <= left < right_t:
                temp.append(m - 1)
                if ((left_t - RGB_ori_norm.shape[0]) <= (left - RGB_ori_norm.shape[0]) < (right_t- RGB_ori_norm.shape[0])) and (0<=(left_t - RGB_ori_norm.shape[0])):
                    temp.append(m - 1 - RGB_ori_norm.shape[0])
                if ((left_t + RGB_ori_norm.shape[0]) <= (left + RGB_ori_norm.shape[0]) < (right_t + RGB_ori_norm.shape[0])) and ((right_t + RGB_ori_norm.shape[0])<=(RGB_ori_norm.shape[0] * RGB_ori_norm.shape[1])):
                    temp.append(m - 1 + RGB_ori_norm.shape[0])
            if left_t <= right < right_t:
                temp.append(m + 1)
                if((left_t- RGB_ori_norm.shape[0]) <= (right - RGB_ori_norm.shape[0]) < (right_t - RGB_ori_norm.shape[0])) and (0<=(left_t - RGB_ori_norm.shape[0])):
                    temp.append(m + 1 - RGB_ori_norm.shape[0])
                if((left_t+ RGB_ori_norm.shape[0]) <= (right + RGB_ori_norm.shape[0]) < (right_t + RGB_ori_norm.shape[0])) and ((right_t + RGB_ori_norm.shape[0])<=(RGB_ori_norm.shape[0] * RGB_ori_norm.shape[1])):
                    temp.append(m + 1 + RGB_ori_norm.shape[0])
            if 0 <= up < (RGB_ori_norm.shape[0] * RGB_ori_norm.shape[1]):
                temp.append(up)
            if 0 <= down < (RGB_ori_norm.shape[0] * RGB_ori_norm.shape[1]):
                temp.append(down)

            graph[m] = temp
            m = m + 1
            temp = []

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    #get train/val/test mask
    test_size = None

    random_state = np.random.RandomState(seed)
    remaining_indices = list(range(labels.shape[0]))
    train_indices = random_state.choice(remaining_indices, 20*labels.shape[1], replace=False)
    train_mask = sample_mask(train_indices, labels.shape[0])

    remaining_indices = np.setdiff1d(remaining_indices, train_indices)
    val_indices = random_state.choice(remaining_indices, 30*labels.shape[1], replace=False)
    val_mask = sample_mask(val_indices, labels.shape[0])

    forbidden_indices = np.concatenate((train_indices, val_indices))

    if test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
        test_mask = sample_mask(test_indices, labels.shape[0])
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_mask = sample_mask(test_indices, labels.shape[0])

    #get y_train/y_val/y_test
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]