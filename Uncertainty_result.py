from gcn.metrics import *
# from gcn.utils import load_data_threshold, load_data, load_data_ood, load_data_adv_nodes
import random
from scipy.special import gamma, digamma

def vacuity_uncertainty(Baye_result):
    # Vacuity uncertainty
    mean = np.mean(Baye_result, axis=0)
    class_num = mean.shape[1]
    alpha = np.exp(mean) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    un_vacuity = class_num / S

    return un_vacuity


def vacuity_uncertainty_edl(mean):
    # Vacuity uncertainty
    # mean = np.mean(Baye_result, axis
    class_num = mean.shape[1]
    alpha = mean + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    un_vacuity = class_num / S

    return un_vacuity

def dissonance_uncertainty(Baye_result):
    mean = np.mean(Baye_result, axis=0)
    evidence = np.exp(mean)
    alpha = np.exp(mean) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    belief = evidence / S
    dis_un = np.zeros_like(S)
    for k in range(belief.shape[0]):
        for i in range(belief.shape[1]):
            bi = belief[k][i]
            term_Bal = 0.0
            term_bj = 0.0
            for j in range(belief.shape[1]):
                if j != i:
                    bj = belief[k][j]
                    term_Bal += bj * Bal(bi, bj)
                    term_bj += bj
            dis_ki = bi * term_Bal / term_bj
            dis_un[k] += dis_ki

    return dis_un


def diff_entropy(Baye_result):
    mean = np.mean(Baye_result, axis=0)
    alpha = np.exp(mean) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    ln_gamma = np.log(gamma(alpha))
    ln_gamma_S = np.log(gamma(S))
    term1 = np.sum(ln_gamma, axis=1, keepdims=True) - ln_gamma_S
    digmma_alpha = digamma(alpha)
    digamma_S = digamma(S)
    term2 = (alpha - 1) * (digmma_alpha - digamma_S)
    term2 = np.sum(term2, axis=1, keepdims=True)
    diff_en = term1 - term2

    return diff_en

def dissonance_uncertainty_edl(Baye_result):
    mean = np.mean(Baye_result, axis=0)
    evidence = mean
    alpha = mean + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    belief = evidence / S
    dis_un = np.zeros_like(S)
    for k in range(belief.shape[0]):
        for i in range(belief.shape[1]):
            bi = belief[k][i]
            term_Bal = 0.0
            term_bj = 0.0
            for j in range(belief.shape[1]):
                if j != i:
                    bj = belief[k][j]
                    term_Bal += bj * Bal(bi, bj)
                    term_bj += bj
            dis_ki = bi * term_Bal / term_bj
            dis_un[k] += dis_ki

    return dis_un

def Bal(b_i, b_j):
    result = 1 - np.abs(b_i - b_j) / (b_i + b_j)
    return result


def entropy_SL(mean):
    class_num = mean.shape[1]
    alpha = np.exp(mean) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    prob = alpha / S
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un

def entropy(pred):
    class_num = pred.shape[1]
    prob = pred
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un

def entropy_softmax(pred):
    class_num = pred.shape[1]
    prob = softmax(pred)
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un

def entropy_edl(pred):
    class_num = pred.shape[1]
    alpha = pred + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    prob = alpha / S
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un

def entropy_dropout(pred):
    mean = []
    for p in pred:
        prob_i = softmax(p)
        mean.append(prob_i)
    mean = np.mean(mean, axis=0)
    class_num = mean.shape[1]
    prob = mean
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un


def aleatoric_dropout(Baye_result):
    al_un = []
    al_class_un = []
    for item in Baye_result:
        un, class_un = entropy_softmax(item)
        al_un.append(un)
        al_class_un.append(class_un)
    ale_un = np.mean(al_un, axis=0)
    ale_class_un = np.mean(al_class_un, axis=0)
    return ale_un, ale_class_un


def epistemic_var(Baye_result):

    E_x2 = np.square(Baye_result)
    E_x2 = np.mean(E_x2, axis=0)
    E2_x = np.mean(Baye_result, axis=0)
    E2_x = np.square(E2_x)
    e_un_class = E_x2 - E2_x
    e_un = np.sum(e_un_class, axis=1, keepdims=True)
    return e_un, e_un_class

def epistemic_var2(Baye_result):
    alpha = np.exp(Baye_result) + 1.0
    S = np.sum(alpha, axis=2, keepdims=True)
    prob = alpha / S

    E_x2 = np.square(prob)
    E_x2 = np.mean(E_x2, axis=0)
    E2_x = np.mean(prob, axis=0)
    E2_x = np.square(E2_x)
    e_un_class = E_x2 - E2_x
    e_un = np.sum(e_un_class, axis=1, keepdims=True)
    return e_un, e_un_class

def softmax(pred):
    ex = np.exp(pred - np.amax(pred, axis=1, keepdims=True))
    prob = ex / np.sum(ex, axis=1, keepdims=True)
    return prob

def total_uncertainty(Baye_result):
    prob_all = []
    class_num = Baye_result[0].shape[1]
    for item in Baye_result:
        alpha = np.exp(item) + 1.0
        S = np.sum(alpha, axis=1, keepdims=True)
        prob = alpha / S
        prob_all.append(prob)
    prob_mean = np.mean(prob_all, axis=0)
    total_class_un = - prob_mean * (np.log(prob_mean) / np.log(class_num))
    total_un = np.sum(total_class_un, axis=1, keepdims=True)
    return total_un, total_class_un


def aleatoric_uncertainty(Baye_result):
    al_un = []
    al_class_un = []
    for item in Baye_result:
        un, class_un = entropy_SL(item)
        al_un.append(un)
        al_class_un.append(class_un)
    ale_un = np.mean(al_un, axis=0)
    ale_class_un = np.mean(al_class_un, axis=0)
    return ale_un, ale_class_un

def gcn_predict(pred):
    mean = np.mean(pred, axis=0)
    gcn_pred = np.argmax(mean, axis=1).reshape([len(np.argmax(mean, axis=1)),1])
    return gcn_pred


def test_uncertainty(Baye_result):
    al_un = []
    al_class_un = []
    Baye_result = Baye_result[:10]
    pp = []
    class_num = Baye_result[0].shape[1]
    for item in Baye_result:
        alpha = np.exp(item) + 1.0
        S = np.sum(alpha, axis=1, keepdims=True)
        prob = alpha / S
        pp.append(prob)
        un, class_un = entropy_SL(item)
        al_un.append(un)
        al_class_un.append(class_un)
    ale_un = np.mean(al_un, axis=0)
    ale_class_un = np.mean(al_class_un, axis=0)
    pp = np.mean(pp, axis=0)

    total_class_un = - pp * (np.log(pp) / np.log(class_num))

    un_e_class = total_class_un - ale_class_un

    return ale_un, ale_class_un

def save_uncertainty():
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets:
        uncertainty = []
        uncertainty_class = []
        Baye_result = np.load("data/baye_500_result_{}.npy".format(dataset))
        un_vacuity = vacuity_uncertainty(Baye_result)
        un_dissonance = dissonance_uncertainty(Baye_result)
        un_total, un_total_class = total_uncertainty(Baye_result)
        un_aleatoric, un_aleatoric_class = aleatoric_uncertainty(Baye_result)
        un_epistemic_class = un_total_class - un_aleatoric_class
        un_epistemic = np.sum(un_epistemic_class, axis=1, keepdims=True)
        un_ep_var, un_ep_var_class = epistemic_var(Baye_result)
        un_ep_var2, un_ep_var_class2 = epistemic_var2(Baye_result)
        uncertainty.append(un_vacuity)
        uncertainty.append(un_dissonance)
        uncertainty.append(un_aleatoric)
        uncertainty.append(un_epistemic)
        uncertainty.append(un_total)
        uncertainty.append(un_ep_var)
        uncertainty.append(un_ep_var2)
        uncertainty_class.append(un_aleatoric_class)
        uncertainty_class.append(un_epistemic_class)
        uncertainty_class.append(un_total_class)
        uncertainty_class.append(un_ep_var_class)
        uncertainty_class.append(un_ep_var_class2)

        np.save("data/uncertainty/baye_500_uncertainty_{}.npy".format(dataset), uncertainty)
        np.save("data/uncertainty/baye_500_uncertainty_{}_class.npy".format(dataset), uncertainty_class)
    return


def get_uncertainty(Baye_result):
    uncertainty = []
    uncertainty_class = []
    un_gcn_pred = gcn_predict(Baye_result)
    un_vacuity = vacuity_uncertainty(Baye_result)
    un_dissonance = dissonance_uncertainty(Baye_result)
    un_total, un_total_class = total_uncertainty(Baye_result)
    un_aleatoric, un_aleatoric_class = aleatoric_uncertainty(Baye_result)
    # un_epistemic = un_total - un_aleatoric
    un_epistemic_class = un_total_class - un_aleatoric_class
    un_epistemic = np.sum(un_epistemic_class, axis=1, keepdims=True)
    un_ep_var, un_ep_var_class = epistemic_var(Baye_result)
    un_ep_var2, un_ep_var_class2 = epistemic_var2(Baye_result)
    un_var3_class = np.var(Baye_result, axis=0)
    un_var3 = np.sum(un_var3_class, axis=1, keepdims=True)
    uncertainty.append(un_vacuity)
    uncertainty.append(un_dissonance)
    uncertainty.append(un_aleatoric)
    uncertainty.append(un_epistemic)
    uncertainty.append(un_total)
    uncertainty.append(un_gcn_pred)
    # uncertainty.append(un_ep_var)
    # uncertainty.append(un_ep_var2)
    # uncertainty.append(un_var3)
    uncertainty_class.append(un_aleatoric_class)
    uncertainty_class.append(un_epistemic_class)
    uncertainty_class.append(un_total_class)
    uncertainty_class.append(un_ep_var_class)
    uncertainty_class.append(un_ep_var_class2)

    return uncertainty


def test_ood(dataset, uncertainties):
    _, _, _, _, _, _, test_mask_all, test_mask = load_data_ood(dataset)
    cdf = []
    for un in uncertainties:
        # un = normalize_un(un)
        un_new = []
        cdf_un = []
        for i in range(len(un)):
            if test_mask[i] == True:
                un_new.append(un[i])
        num_test = len(un_new)
        for threshold in np.arange(0.0, 1.01, 0.02):
            ood_num = 0.0
            for uni in un_new:
                if uni < threshold:
                    ood_num += 1.0
            cdf_t = ood_num / num_test
            cdf_un.append(cdf_t)
        cdf.append(cdf_un)
    return cdf

def get_tp(pred, label):
    tp = 0.0
    for i in range(len(pred)):
        if pred[i] and label[i]:
            tp += 1
    return tp

def test_ood_auroc(dataset, uncertainties):
    _, _, _, _, _, _, test_mask_all, test_mask = load_data_ood(dataset)
    tpr_data = []
    fpr_data = []
    train_num = len(test_mask_all) - np.sum(test_mask_all)
    label = test_mask[train_num:]
    tp_fn = np.sum(label)
    fp_tn = len(label) - tp_fn
    for un in uncertainties:
        un_new = un[train_num:]
        un_new = normalize_un(un_new)
        tpr = []
        fpr = []
        for threshold in np.arange(0.0, 1.00001, 0.01):
            ood_pred = np.zeros_like(label, dtype=bool)
            for i in range(len(un_new)):
                if un_new[i] >= threshold:
                    ood_pred[i] = True
            tp = get_tp(ood_pred, label)
            fp = np.sum(ood_pred) - tp
            tpr_t = tp / tp_fn
            fpr_t = fp / fp_tn
            tpr.append(tpr_t)
            fpr.append(fpr_t)
        tpr_data.append(tpr)
        fpr_data.append(fpr)
    return tpr_data, fpr_data


def test_ood_auroc2(dataset, uncertainties):
    _, _, _, _, _, _, test_mask_all, test_mask = load_data_ood(dataset)
    tpr_data = []
    fpr_data = []
    test_index = random_test_(1000, test_mask_all)
    for test_index_i in test_index:
        tpr_i = []
        fpr_i = []
        label = []
        for i in test_index_i:
            if test_mask[i]:
                label.append(1.0)
            else:
                label.append(0.0)
        tp_fn = np.sum(label)
        fp_tn = len(label) - tp_fn
        for un in uncertainties:
            un_new = []
            for i in test_index_i:
                un_new.append(un[i])
            un_new = normalize_un(un_new)
            tpr = []
            fpr = []
            for threshold in np.arange(0.0, 1.000001, 0.01):
                ood_pred = np.zeros_like(label, dtype=bool)
                for i in range(len(un_new)):
                    if un_new[i] >= threshold:
                        ood_pred[i] = True
                tp = get_tp(ood_pred, label)
                fp = np.sum(ood_pred) - tp
                tpr_t = tp / tp_fn
                fpr_t = fp / fp_tn
                tpr.append(tpr_t)
                fpr.append(fpr_t)
            tpr_i.append(tpr)
            fpr_i.append(fpr)
        tpr_data.append(tpr_i)
        fpr_data.append(fpr_i)
    return tpr_data, fpr_data


def random_test_(ratio_num, test_mask_all):
    random.seed(123)
    train_num = len(test_mask_all) - np.sum(test_mask_all)
    test_index = []
    for i in range(50):
        test_index_i = random.sample(range(int(train_num), len(test_mask_all)), ratio_num)
        test_index.append(test_index_i)
    return test_index

def test_ood_aupr(dataset, uncertainties):
    _, _, _, _, _, _, test_mask_all, test_mask = load_data_ood(dataset)
    recall_data = []
    precision_data = []
    train_num = len(test_mask_all) - np.sum(test_mask_all)
    label = test_mask[train_num:]
    tp_fn = np.sum(label)
    fp_tn = len(label) - tp_fn
    for un in uncertainties:
        un = normalize_un(un)
        un_new = un[train_num:]
        recall_t = []
        precision_t = []
        for threshold in np.arange(0.0, 0.8, 0.01):
            ood_pred = np.zeros_like(label, dtype=bool)
            for i in range(len(un_new)):
                if un_new[i] >= threshold:
                    ood_pred[i] = True
            tp = get_tp(ood_pred, label)
            fp = np.sum(ood_pred) - tp
            recall = tp / tp_fn
            precision = tp / np.sum(ood_pred)
            recall_t.append(recall)
            precision_t.append(precision)
        for threshold in np.arange(0.8, 1.00000001, 0.01):
            ood_pred = np.zeros_like(label, dtype=bool)
            for i in range(len(un_new)):
                if un_new[i] >= threshold:
                    ood_pred[i] = True
            tp = get_tp(ood_pred, label)
            fp = np.sum(ood_pred) - tp
            recall = tp / tp_fn
            precision = tp / np.sum(ood_pred)
            recall_t.append(recall)
            precision_t.append(precision)
        recall_data.append(recall_t)
        precision_data.append(precision_t)
    return recall_data, precision_data


def test_ood_aupr2(dataset, uncertainties):
    _, _, _, _, _, _, test_mask_all, test_mask = load_data_ood(dataset)
    recall_data = []
    precision_data = []
    test_index = random_test_(1000, test_mask_all)
    for test_index_i in test_index:
        recall_i = []
        precision_i = []
        label = []
        for i in test_index_i:
            if test_mask[i]:
                label.append(1.0)
            else:
                label.append(0.0)

        tp_fn = np.sum(label)
        fp_tn = len(label) - tp_fn
        for un in uncertainties:
            un_new = []
            for i in test_index_i:
                un_new.append(un[i])
            un_new = normalize_un(un_new)
            recall_t = []
            precision_t = []
            for threshold in np.arange(0.0, 0.8, 0.01):
                ood_pred = np.zeros_like(label, dtype=bool)
                for i in range(len(un_new)):
                    if un_new[i] >= threshold:
                        ood_pred[i] = True
                tp = get_tp(ood_pred, label)
                fp = np.sum(ood_pred) - tp
                recall = tp / tp_fn
                precision = tp / np.sum(ood_pred)
                recall_t.append(recall)
                precision_t.append(precision)
            for threshold in np.arange(0.8, 1.00000001, 0.01):
                ood_pred = np.zeros_like(label, dtype=bool)
                for i in range(len(un_new)):
                    if un_new[i] >= threshold:
                        ood_pred[i] = True
                tp = get_tp(ood_pred, label)
                fp = np.sum(ood_pred) - tp
                recall = tp / tp_fn
                precision = tp / np.sum(ood_pred)
                recall_t.append(recall)
                precision_t.append(precision)
            recall_i.append(recall_t)
            precision_i.append(precision_t)
        recall_data.append(recall_i)
        precision_data.append(precision_i)
    return recall_data, precision_data


def test_ood_aupr3():
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets:
        _, _, _, _, _, _, test_mask_all, test_mask = load_data_ood(dataset)
        test_num_all = np.sum(test_mask_all)
        # Baye_result = np.load("data/baye_500_result_{}.npy".format(dataset))
        # mean = np.mean(Baye_result, axis=0)
        uncertainties = np.load("data/ood/result/uncertainty_all_{}_ood.npy".format(dataset))
        prediction = test_mask

        random.seed(123)
        train_num = len(test_mask_all) - test_num_all
        test_index = []
        for i in range(50):
            test_index_i = random.sample(range(int(train_num), len(test_mask)), 1000)
            test_index.append(test_index_i)
        recall_data = []
        precision_data = []
        for index in test_index:
            prediction_i = []

            recall_index = []
            precision_index = []
            for i in index:
                prediction_i.append(prediction[i])

            tp_fn = np.sum(prediction_i)
            fp_tn = len(prediction_i) - tp_fn

            for uncertainty in uncertainties:
                un_i = []
                for i in index:
                    un_i.append(uncertainty[i])
                un_i = np.array(un_i) * (-1.0)
                recall_un = []
                precision_un = []
                for test_ratio in np.arange(0.05, 1.0001, 0.05):
                    recall_num = int(tp_fn * test_ratio)
                    ratio_num = recall_num
                    mask_un = ratio_mask2(ratio_num, un_i)
                    ood_pred = mask_un * prediction_i
                    tp = float(np.sum(ood_pred))
                    while tp < recall_num:
                        ratio_num += 1
                        mask_un = ratio_mask2(ratio_num, un_i)
                        ood_pred = mask_un * prediction_i
                        tp = np.sum(ood_pred)
                    recall = float(tp) / tp_fn
                    precision = float(tp) / np.sum(mask_un)
                    print(np.sum(mask_un))
                    recall_un.append(recall)
                    precision_un.append(precision)
                recall_index.append(recall_un)
                precision_index.append(precision_un)
            recall_data.append(recall_index)
            precision_data.append(precision_index)
        np.save("data/ood/result/test_ratio/recall_{}_ood0.05_1000.npy".format(dataset), recall_data)
        np.save("data/ood/result/test_ratio/precision_{}_ood0.05_1000.npy".format(dataset), precision_data)
    return


def test_ood_auroc3():
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets:
        _, _, _, _, _, _, test_mask_all, test_mask = load_data_ood(dataset)
        test_num_all = np.sum(test_mask_all)
        # Baye_result = np.load("data/baye_500_result_{}.npy".format(dataset))
        # mean = np.mean(Baye_result, axis=0)
        uncertainties = np.load("data/ood/result/uncertainty_all_{}_ood.npy".format(dataset))
        prediction = test_mask

        random.seed(123)
        train_num = len(test_mask_all) - test_num_all
        test_index = []
        for i in range(50):
            test_index_i = random.sample(range(int(train_num), len(test_mask)), 1000)
            test_index.append(test_index_i)
        recall_data = []
        precision_data = []
        for index in test_index:
            prediction_i = []

            recall_index = []
            precision_index = []
            for i in index:
                prediction_i.append(prediction[i])

            tp_fn = np.sum(prediction_i)
            fp_tn = len(prediction_i) - tp_fn

            for uncertainty in uncertainties:
                un_i = []
                for i in index:
                    un_i.append(uncertainty[i])
                un_i = np.array(un_i) * (-1.0)
                recall_un = []
                precision_un = []

                ratio_num = 1
                mask_un = ratio_mask2(ratio_num, un_i)
                ood_pred = mask_un * prediction_i
                tp = float(np.sum(ood_pred))
                fp = np.sum(mask_un) - tp
                while fp < 0.5:
                    ratio_num += 1
                    mask_un = ratio_mask2(ratio_num, un_i)
                    ood_pred = mask_un * prediction_i
                    tp = np.sum(ood_pred)
                    fp = np.sum(mask_un) - tp
                tp = float(np.sum(ood_pred))
                tpr_t = tp / tp_fn
                recall_un.append(0.0)
                precision_un.append(tpr_t)


                for test_ratio in np.arange(0.05, 1.0001, 0.05):
                    fptn_num = int(fp_tn * test_ratio)
                    ratio_num = fptn_num
                    mask_un = ratio_mask2(ratio_num, un_i)
                    ood_pred = mask_un * prediction_i
                    tp = float(np.sum(ood_pred))
                    fp = np.sum(mask_un) - tp
                    while fp < fptn_num:
                        ratio_num += 1
                        mask_un = ratio_mask2(ratio_num, un_i)
                        ood_pred = mask_un * prediction_i
                        tp = np.sum(ood_pred)
                        fp = np.sum(mask_un) - tp
                    tp = float(np.sum(ood_pred))
                    fp = np.sum(mask_un) - tp
                    tpr_t = tp / tp_fn
                    fpr_t = fp / fp_tn
                    recall_un.append(fpr_t)
                    precision_un.append(tpr_t)
                recall_index.append(recall_un)
                precision_index.append(precision_un)
            recall_data.append(recall_index)
            precision_data.append(precision_index)
        np.save("data/ood/result/test_ratio/fpr_{}_ood0.05_1000.npy".format(dataset), recall_data)
        np.save("data/ood/result/test_ratio/tpr_{}_ood0.05_1000.npy".format(dataset), precision_data)
    return

def test_ood_id(dataset, uncertainties):
    _, _, _, _, _, train_mask, _, test_mask = load_data_ood(dataset)
    cdf_dataset = []
    for un in uncertainties:
        un = normalize_un(un)
        cdf = []
        un_ood = []
        un_id = []
        un_train = []
        cdf_ood = []
        cdf_id = []
        cdf_train = []
        for i in range(len(un)):
            if test_mask[i] == True:
                un_ood.append(un[i])
            else:
                un_id.append(un[i])
            if train_mask[i] == True:
                un_train.append(un[i])
        num_ood = len(un_ood)
        num_id = len(un_id)
        num_train = len(un_train)
        for threshold in np.arange(0.0, 1.01, 0.02):
            ood_num = 0.0
            id_num = 0.0
            train_num = 0.0
            for uni in un_ood:
                if uni < threshold:
                    ood_num += 1.0
            cdf_t = ood_num / num_ood
            cdf_ood.append(cdf_t)

            for unid in un_id:
                if unid < threshold:
                    id_num += 1.0
            cdf_t = id_num / num_id
            cdf_id.append(cdf_t)

            for unitrain in un_train:
                if unitrain < threshold:
                    train_num += 1.0
            cdf_t = train_num / num_train
            cdf_train.append(cdf_t)

        cdf.append(cdf_ood)
        cdf.append(cdf_id)
        cdf.append(cdf_train)

        cdf_dataset.append(cdf)
    return cdf_dataset

def test_ood_gcn():
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets:
        gcn_pred = np.load("data/ood/gcn_{}_ood.npy".format(dataset))
        un_total, _ = entropy(gcn_pred)
        _, _, _, _, _, train_mask, _, test_mask = load_data_ood(dataset)
        # un = normalize_un(un)
        cdf = []
        un_ood = []
        un_id = []
        un_train = []
        cdf_ood = []
        cdf_id = []
        cdf_train = []
        for i in range(len(un_total)):
            if test_mask[i] == True:
                un_ood.append(un_total[i])
            else:
                un_id.append(un_total[i])
            if train_mask[i] == True:
                un_train.append(un_total[i])
        num_ood = len(un_ood)
        num_id = len(un_id)
        num_train = len(un_train)
        for threshold in np.arange(0.0, 1.01, 0.02):
            ood_num = 0.0
            id_num = 0.0
            train_num = 0.0
            for uni in un_ood:
                if uni < threshold:
                    ood_num += 1.0
            cdf_t = ood_num / num_ood
            cdf_ood.append(cdf_t)

            for unid in un_id:
                if unid < threshold:
                    id_num += 1.0
            cdf_t = id_num / num_id
            cdf_id.append(cdf_t)

            for unitrain in un_train:
                if unitrain < threshold:
                    train_num += 1.0
            cdf_t = train_num / num_train
            cdf_train.append(cdf_t)

        cdf.append(cdf_ood)
        cdf.append(cdf_id)
        cdf.append(cdf_train)
        np.save("data/ood/gcn_cdf_{}_ood.npy".format(dataset), cdf)
    return


def test_ood_edl():
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets:
        edl_pred = np.load("data/ood/edl_{}_ood.npy".format(dataset))
        un_total, _ = entropy_edl(edl_pred)
        un_v = vacuity_uncertainty_edl(edl_pred)
        _, _, _, _, _, train_mask, _, test_mask = load_data_ood(dataset)
        # un = normalize_un(un)
        cdf = []
        un_ood = []
        un_id = []
        un_train = []
        cdf_ood = []
        cdf_id = []
        cdf_train = []
        for i in range(len(un_total)):
            if test_mask[i] == True:
                un_ood.append(un_total[i])
            else:
                un_id.append(un_total[i])
            if train_mask[i] == True:
                un_train.append(un_total[i])
        num_ood = len(un_ood)
        num_id = len(un_id)
        num_train = len(un_train)
        for threshold in np.arange(0.0, 1.01, 0.02):
            ood_num = 0.0
            id_num = 0.0
            train_num = 0.0
            for uni in un_ood:
                if uni < threshold:
                    ood_num += 1.0
            cdf_t = ood_num / num_ood
            cdf_ood.append(cdf_t)

            for unid in un_id:
                if unid < threshold:
                    id_num += 1.0
            cdf_t = id_num / num_id
            cdf_id.append(cdf_t)

            for unitrain in un_train:
                if unitrain < threshold:
                    train_num += 1.0
            cdf_t = train_num / num_train
            cdf_train.append(cdf_t)

        cdf.append(cdf_ood)
        cdf.append(cdf_id)
        cdf.append(cdf_train)
        np.save("data/ood/edl_cdf_{}_ood.npy".format(dataset), cdf)
    return


def test_ood_dropout():
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets:
        pred = np.load("data/ood/dropout_{}_ood.npy".format(dataset))
        un_total, _ = entropy_dropout(pred)
        _, _, _, _, _, train_mask, _, test_mask = load_data_ood(dataset)
        # un = normalize_un(un)
        cdf = []
        un_ood = []
        un_id = []
        un_train = []
        cdf_ood = []
        cdf_id = []
        cdf_train = []
        for i in range(len(un_total)):
            if test_mask[i] == True:
                un_ood.append(un_total[i])
            else:
                un_id.append(un_total[i])
            if train_mask[i] == True:
                un_train.append(un_total[i])
        num_ood = len(un_ood)
        num_id = len(un_id)
        num_train = len(un_train)
        for threshold in np.arange(0.0, 1.01, 0.02):
            ood_num = 0.0
            id_num = 0.0
            train_num = 0.0
            for uni in un_ood:
                if uni < threshold:
                    ood_num += 1.0
            cdf_t = ood_num / num_ood
            cdf_ood.append(cdf_t)

            for unid in un_id:
                if unid < threshold:
                    id_num += 1.0
            cdf_t = id_num / num_id
            cdf_id.append(cdf_t)

            for unitrain in un_train:
                if unitrain < threshold:
                    train_num += 1.0
            cdf_t = train_num / num_train
            cdf_train.append(cdf_t)

        cdf.append(cdf_ood)
        cdf.append(cdf_id)
        cdf.append(cdf_train)
        np.save("data/ood/dropout_cdf_{}_ood.npy".format(dataset), cdf)
    return

def load_uncertainty(dataset):
    uncertainty = np.load("data/uncertainty/baye_500_uncertainty_{}.npy".format(dataset))
    uncertainty_class = np.load("data/uncertainty/baye_500_uncertainty_{}_class.npy".format(dataset))
    return uncertainty, uncertainty_class


def normalize_un(un):
    max_value = np.max(un)
    min_value = np.min(un)
    un_nor = (un - min_value) / (max_value - min_value)
    return un_nor


def uncertainty_threshold():
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets:
        _, _, _, _, y_test, train_mask, _, _, labels = load_data_threshold(dataset)
        Baye_result = np.load("data/baye_500_result_{}.npy".format(dataset))
        mean = np.mean(Baye_result, axis=0)
        uncertainties = np.load("data/uncertainty/baye_500_uncertainty_{}.npy".format(dataset))
        acc_un_data = []
        for uncertainty in uncertainties:
            uncertainty_nor = normalize_un(uncertainty)
            acc_un = []
            for threshold in np.arange(0.0, 1.01, 0.05):
                mask_un = 1 - np.array(train_mask)
                for i in range(len(uncertainty_nor)):
                    if uncertainty_nor[i] > threshold:
                        mask_un[i] = False
                acc = masked_accuracy_numpy(mean, labels, mask_un)
                acc_un.append(acc)
            acc_un_data.append(acc_un)
        np.save("data/uncertainty/baye_500_threshold_result_{}_0.05_moretest.npy".format(dataset), acc_un_data)
    return


def uncertainty_test_ratio():
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets:
        _, _, _, _, y_test, train_mask, _, test_mask, labels = load_data_threshold(dataset)
        test_mask = 1 - np.array(train_mask)
        test_num_all = np.sum(test_mask)
        Baye_result = np.load("data/baye_500_result_{}.npy".format(dataset))
        mean = np.mean(Baye_result, axis=0)
        uncertainties = np.load("data/uncertainty/baye_500_uncertainty_{}.npy".format(dataset))
        acc_un_data = []
        for uncertainty in uncertainties:
            # uncertainty_nor = normalize_un(uncertainty)
            acc_un = []
            for test_ratio in np.arange(0.05, 1.01, 0.05):
                ratio_num = int(test_num_all * test_ratio)

                mask_un = ratio_mask(ratio_num, train_mask, uncertainty)

                acc = masked_accuracy_numpy(mean, labels, mask_un)
                acc_un.append(acc)
            acc_un_data.append(acc_un)
        # random choose
        acc_un = []
        for test_ratio in np.arange(0.05, 1.01, 0.05):
            ratio_num = int(test_num_all * test_ratio)

            acc_random = []
            for k in range(20):
                mask_un = random_mask(ratio_num, train_mask)
                acc_rand = masked_accuracy_numpy(mean, labels, mask_un)
                acc_random.append(acc_rand)
            acc_un.append(np.mean(acc_random))
        acc_un_data.append(acc_un)

        np.save("data/uncertainty/test_ratio_{}_0.05.npy".format(dataset), acc_un_data)
    return


def uncertainty_test_ratio_gat(dataset, Baye_result, uncertainties):
    _, _, _, _, y_test, train_mask, _, test_mask, labels = load_data_threshold(dataset)
    test_mask = 1 - np.array(train_mask)
    test_num_all = np.sum(test_mask)
    # Baye_result = np.load("/network/rit/lab/ceashpc/sharedata_shu/hidden/Baye_gat_result_{}.npy".format(dataset))
    mean = np.mean(Baye_result, axis=0)
    # uncertainties = get_uncertainty(Baye_result)
    acc_un_data = []
    for uncertainty in uncertainties:
        # uncertainty_nor = normalize_un(uncertainty)
        acc_un = []
        for test_ratio in np.arange(0.05, 1.01, 0.05):
            ratio_num = int(test_num_all * test_ratio)

            mask_un = ratio_mask(ratio_num, train_mask, uncertainty)

            acc = masked_accuracy_numpy(mean, labels, mask_un)
            acc_un.append(acc)
        acc_un_data.append(acc_un)
    # random choose
    acc_un = []
    for test_ratio in np.arange(0.05, 1.01, 0.05):
        ratio_num = int(test_num_all * test_ratio)

        acc_random = []
        for k in range(20):
            mask_un = random_mask(ratio_num, train_mask)
            acc_rand = masked_accuracy_numpy(mean, labels, mask_un)
            acc_random.append(acc_rand)
        acc_un.append(np.mean(acc_random))
    acc_un_data.append(acc_un)
    # np.save("data/uncertainty/test_ratio_{}_0.05.npy".format(dataset), acc_un_data)
    return acc_un_data

def random_mask(ratio_num, train_mask):
    train_num = np.sum(train_mask)
    train_mask_s = np.array(train_mask[:train_num])
    test_mask_s = np.array(train_mask[train_num:])
    test_index = random.sample(range(len(test_mask_s)), ratio_num)
    for i in test_index:
        test_mask_s[i] = True
    mask_un = np.concatenate((1 - np.array(train_mask_s), test_mask_s), axis=0)
    return np.array(mask_un, dtype=bool)

def ratio_mask(ratio_num, train_mask, uncertainty):
    train_num = np.sum(train_mask)
    train_mask_s = np.array(train_mask[:train_num])
    test_mask_s = np.array(train_mask[train_num:])
    uncertainty_s = uncertainty[train_num:]
    test_index = np.argpartition(uncertainty_s, range(ratio_num), axis=0)
    test_index = test_index[:ratio_num]
    for i in test_index:
        test_mask_s[i] = True
    mask_un = np.concatenate((1 - np.array(train_mask_s), test_mask_s), axis=0)
    return np.array(mask_un, dtype=bool)

def uncertainty_test_ratio2():
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets:
        _, _, _, _, y_test, train_mask, _, test_mask, labels = load_data_threshold(dataset)
        test_num_all = np.sum(test_mask)
        Baye_result = np.load("data/baye_500_result_{}.npy".format(dataset))
        mean = np.mean(Baye_result, axis=0)
        uncertainties = get_uncertainty(Baye_result)
        # uncertainties = np.load("data/uncertainty/our_model/uncertainty_all_{}.npy".format(dataset))
        prediction = np.equal(np.argmax(mean, 1), np.argmax(labels, 1))

        random.seed(123)
        train_num = np.sum(train_mask)
        test_index = []
        for i in range(50):
            test_index_i = random.sample(range(int(train_num), len(test_mask)), 1000)
            test_index.append(test_index_i)
        acc_un_data = []
        for index in test_index:
            prediction_i = []

            acc_un_index = []
            for i in index:
                prediction_i.append(prediction[i])

            for uncertainty in uncertainties:
                un_i = []
                for i in index:
                    un_i.append(uncertainty[i])
                # uncertainty_nor = normalize_un(uncertainty)
                acc_un = []
                for test_ratio in np.arange(0.05, 1.000001, 0.05):
                    ratio_num = int(test_num_all * test_ratio)

                    mask_un = ratio_mask2(ratio_num, un_i)

                    acc = masked_accuracy_co(prediction_i, mask_un)
                    acc_un.append(acc)
                acc_un_index.append(acc_un)
            # random choose
            acc_un = []
            for test_ratio in np.arange(0.05, 1.000001, 0.05):
                ratio_num = int(test_num_all * test_ratio)

                acc_random = []
                for k in range(20):
                    mask_un = random_mask2(ratio_num, prediction_i)
                    acc_rand = masked_accuracy_co(prediction_i, mask_un)
                    acc_random.append(acc_rand)
                acc_un.append(np.mean(acc_random))
            acc_un_index.append(acc_un)
            acc_un_data.append(acc_un_index)
        np.save("data/uncertainty/test_ratio/test_ratio_{}_1000.npy".format(dataset), acc_un_data)
    return


def uncertainty_test_ratio_pr():
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets:
        _, _, _, _, y_test, train_mask, _, test_mask, labels = load_data_threshold(dataset)
        test_num_all = np.sum(test_mask)
        Baye_result = np.load("data/baye_500_result_{}.npy".format(dataset))
        mean = np.mean(Baye_result, axis=0)
        uncertainties = np.load("data/uncertainty/baye_500_uncertainty_{}.npy".format(dataset))
        prediction = np.equal(np.argmax(mean, 1), np.argmax(labels, 1))

        random.seed(123)
        train_num = np.sum(train_mask)
        test_index = []
        for i in range(50):
            test_index_i = random.sample(range(int(train_num), len(test_mask)), 1000)
            test_index.append(test_index_i)
        acc_un_data = []
        recall_data = []
        precision_data = []
        for index in test_index:
            prediction_i = []

            acc_un_index = []
            recall_index = []
            precision_index = []
            for i in index:
                prediction_i.append(prediction[i])

            tp_fn = np.sum(prediction_i)
            fp_tn = len(prediction_i) - tp_fn

            for uncertainty in uncertainties:
                un_i = []
                for i in index:
                    un_i.append(uncertainty[i])
                # uncertainty_nor = normalize_un(uncertainty)
                acc_un = []
                recall_un = []
                precision_un = []
                for test_ratio in np.arange(0.02, 1.0001, 0.02):
                    recall_num = int(tp_fn * test_ratio)
                    ratio_num = recall_num
                    mask_un = ratio_mask2(ratio_num, un_i)
                    ood_pred = mask_un * prediction_i
                    tp = float(np.sum(ood_pred))
                    while tp < recall_num:
                        ratio_num += 1
                        mask_un = ratio_mask2(ratio_num, un_i)
                        ood_pred = mask_un * prediction_i
                        tp = np.sum(ood_pred)
                    recall = float(tp) / tp_fn
                    precision = float(tp) / np.sum(mask_un)
                    acc = masked_accuracy_co(prediction_i, mask_un)
                    acc_un.append(acc)
                    recall_un.append(recall)
                    precision_un.append(precision)
                acc_un_index.append(acc_un)
                recall_index.append(recall_un)
                precision_index.append(precision_un)
            acc_un_data.append(acc_un_index)
            recall_data.append(recall_index)
            precision_data.append(precision_index)
        # np.save("data/uncertainty/test_ratio_{}_1000.npy".format(dataset), acc_un_data)
        np.save("data/uncertainty/recall_{}_0.02_1000.npy".format(dataset), recall_data)
        np.save("data/uncertainty/precision_{}_0.02_1000.npy".format(dataset), precision_data)
    return

def random_mask2(ratio_num, prediction_i):
    test_index = random.sample(range(len(prediction_i)), ratio_num)
    mask = np.zeros_like(prediction_i, dtype=bool)
    for i in test_index:
        mask[i] = True
    return mask

def ratio_mask2(ratio_num, uncertainty):
    mask = np.zeros_like(uncertainty, dtype=bool)
    mask = np.reshape(mask, [-1])
    test_index = np.argpartition(uncertainty, range(ratio_num), axis=0)
    test_index = test_index[:ratio_num]
    for i in test_index:
        mask[i] = True
    return mask

def get_un_dropout(pred):
    un = []
    dr_entroy, dr_entroy_class = entropy_dropout(pred)
    dr_ale, dr_ale_clsss = aleatoric_dropout(pred)
    dr_eps_class = dr_entroy_class - dr_ale_clsss
    dr_eps = np.sum(dr_eps_class, axis=1, keepdims=True)
    un.append(dr_entroy)
    un.append(dr_ale)
    un.append(dr_eps)
    return un

def get_un_all_ood():
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets[:]:
        un_all = []
        Baye_result = np.load("data/ood/baye_500_result_{}_ood.npy".format(dataset))
        uncertainty = get_uncertainty(Baye_result)

        dis = np.load("data/ood/distribution/dis_uncertainty_{}_ood.npy".format(dataset))

        gcn_pred = np.load("data/ood/gcn_{}_ood.npy".format(dataset))
        un_gcn, _ = entropy(gcn_pred)

        pred = np.load("data/ood/dropout_{}_ood.npy".format(dataset))
        dr_entroy, dr_entroy_class = entropy_dropout(pred)
        dr_ale, dr_ale_clsss = aleatoric_dropout(pred)
        dr_eps_class = dr_entroy_class - dr_ale_clsss
        dr_eps = np.sum(dr_eps_class, axis=1, keepdims=True)
        for uni in uncertainty:
            un_all.append(uni)
        un_all.append(dis)
        un_all.append(un_gcn)
        un_all.append(dr_entroy)
        un_all.append(dr_ale)
        un_all.append(dr_eps)

        np.save("data/ood/1000/uncertainty_all_{}_ood_1000.npy".format(dataset), un_all)
    return


def get_un_all():
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets[:]:
        un_all = []
        Baye_result = np.load("data/baye_500_result_{}.npy".format(dataset))
        uncertainty = get_uncertainty(Baye_result)
        # np.save("data/uncertainty/our_model/uncertainty_{}.npy".format(dataset), uncertainty)
        # un = np.load("/network/rit/lab/ceashpc/xujiang/project/Uncertainty_aware/gcn/data/uncertainty/uncertainty_all_cora.npy")
        gcn_pred = np.load("data/result/gcn_{}.npy".format(dataset))
        un_gcn, _ = entropy(gcn_pred)

        pred = np.load("data/result/dropout_{}.npy".format(dataset))
        dr_entroy, _ = entropy_dropout(pred)
        dr_ale, _ = aleatoric_dropout(pred)
        dr_eps = dr_entroy - dr_ale

        for uni in uncertainty:
            un_all.append(uni)
        un_all.append(un_gcn)
        un_all.append(dr_entroy)
        un_all.append(dr_ale)
        un_all.append(dr_eps)

        np.save("data/uncertainty/our_model/uncertainty_all_{}.npy".format(dataset), un_all)
    return


def get_un_adv(method, node):
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets[0:1]:

        Baye_result_all = np.load("data/adv/{}_{}_{}.npy".format(method, dataset, node))
        gcn_pred_all = np.load("data/adv/{}_{}_gcn_{}.npy".format(method, dataset, node))
        pred_all = np.load("data/adv/{}_{}_dropout_{}.npy".format(method, dataset, node))
        un_p = []
        for i in range(len(gcn_pred_all)):
            un_all = []
            Baye_result = Baye_result_all[i]
            uncertainty = get_uncertainty(Baye_result)


            gcn_pred = gcn_pred_all[i]
            un_gcn, _ = entropy(gcn_pred)

            pred = pred_all[i]
            dr_entroy, _ = entropy_dropout(pred)
            dr_ale, _ = aleatoric_dropout(pred)
            dr_eps = dr_entroy - dr_ale

            for uni in uncertainty:
                un_all.append(uni)
            un_all.append(un_gcn)
            un_all.append(dr_entroy)
            un_all.append(dr_ale)
            un_all.append(dr_eps)
            un_p.append(un_all)
        np.save("data/adv/{}_{}_uncertainty_{}.npy".format(method, dataset, node), un_p)
    return

def mask_mean(un, mask):
    aa = un * mask
    uu_ = np.array(un)
    mask = np.reshape(mask, [-1, 1])
    mask = np.asarray(mask, dtype=np.float32)
    mask /= np.mean(mask)
    uu_ *= mask
    return np.mean(uu_)

def value_un(method, node):
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets[0:1]:
        _, _, _, _, _, train_mask, val_mask, test_mask = load_data_adv_nodes(dataset, 1, node)
        uncertainty = np.load("data/adv/{}_{}_uncertainty_{}.npy".format(method, dataset, node))
        un_value = []
        un_value_nor = []
        for up in uncertainty:
            up_mean = []
            up_nor = []
            for ui in up:
                ui_nor = normalize_un(ui)
                train_un = mask_mean(ui, train_mask)
                val_un = mask_mean(ui, val_mask)
                test_un = mask_mean(ui, test_mask)

                train_un_nor = mask_mean(ui_nor, train_mask)
                val_un_nor = mask_mean(ui_nor, val_mask)
                test_un_nor = mask_mean(ui_nor, test_mask)
                up_mean.append([train_un, val_un, test_un])
                up_nor.append([train_un_nor, val_un_nor, test_un_nor])
            un_value.append(up_mean)
            un_value_nor.append(up_nor)
        np.save("data/adv/{}_{}_uncertainty_mean_{}.npy".format(method, dataset, node), un_value)
        np.save("data/adv/{}_{}_uncertainty_mean_nor_{}.npy".format(method, dataset, node), un_value_nor)
    return


def value_un2(method):
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets[1:2]:
        _, _, _, _, _, train_mask, val_mask, test_mask = load_data(dataset)
        uncertainty = np.load("data/adv/feat/{}_{}_uncertainty.npy".format(method, dataset))
        un_value = []
        un_value_nor = []
        for up in uncertainty:
            up_mean = []
            up_nor = []
            for ui in up:
                ui_nor = normalize_un(ui)
                for i in range(4):
                    uuu = np.array(ui[500*(i+1):500*(i+2)])
                    mean_u = np.mean(uuu)
                train_un = mask_mean(ui, train_mask)
                val_un = mask_mean(ui, val_mask)
                test_un = mask_mean(ui, test_mask)

                train_un_nor = mask_mean(ui_nor, train_mask)
                val_un_nor = mask_mean(ui_nor, val_mask)
                test_un_nor = mask_mean(ui_nor, test_mask)
                up_mean.append([train_un, val_un, test_un])
                up_nor.append([train_un_nor, val_un_nor, test_un_nor])
            un_value.append(up_mean)
            un_value_nor.append(up_nor)
        np.save("data/adv/{}_{}_uncertainty_mean.npy".format(method, dataset), un_value)
        np.save("data/adv/{}_{}_uncertainty_mean_nor.npy".format(method, dataset), un_value_nor)
    return