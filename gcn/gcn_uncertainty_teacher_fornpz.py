from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

import os
import sys

sys.path.append(os.path.abspath('/network/rit/lab/ceashpc/shuhu/gcn_uncertainty_cotrain'))

from gcn.utils import *
from gcn.models import GCN, MLP, GCN_uncertainty, GCN_uncertainty_teacher, GCN_EDL, GCN_uncertainty_teacher_cotrain
from gcn.metrics import *
from Load_npz import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora_full', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed', 'nell.0.001', 'ms_academic_phy', 'ms_academic_cs', 'amazon_electronics_photo', 'amazon_electronics_computers', 'cora_full'
flags.DEFINE_string('model', 'gcn_un_t', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense', 'gcn_un_t', 'gcn_edl', 'gcn_un_t_cotrain'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.') #64
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).') # 0.1
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.') #1e-5
flags.DEFINE_float('weight_teacher', 1.0, 'Weight for teacher KL.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


print("Dataset : ", FLAGS.dataset)
print("Model : ", FLAGS.model)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

random_test = []  # random result
Bayesian_result = []  # Bayesian result
Bayesian_result_predict = []

# Define model evaluation function
def evaluate(features, support, labels, mask, epoch, gcn_pred, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict_un_teacher(features, support, labels, mask, epoch, gcn_pred, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

gcn_pred = np.load("/network/rit/lab/ceashpc/sharedata_shu/gcn_{}_10_random_split_new.npy".format(FLAGS.dataset))

for k in range(10):
    # Load data
    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_nell(FLAGS.dataset)
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_npz_data(FLAGS.dataset, seed+k*100)
    # gcn_pred = np.load("data/gat_{}_10.npy".format(FLAGS.dataset))
    # gcn_pred = np.load("/network/rit/lab/ceashpc/sharedata_shu/gcn_{}_10_random_split_new.npy".format(FLAGS.dataset))
    # gcn_pred = []
    # for i in range(10):
    #     gcn_pred_i = np.load("/network/rit/lab/ceashpc/sharedata_shu/gat_baseline/gat_{}_{}.npy".format(FLAGS.dataset, i+1))
    #     gcn_pred.append(gcn_pred_i)



    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    elif FLAGS.model == 'gcn_un_t':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN_uncertainty_teacher
    elif FLAGS.model == 'gcn_un_t_cotrain':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN_uncertainty_teacher_cotrain
    elif FLAGS.model == 'gcn_edl':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN_EDL
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'annealing_step': tf.placeholder(tf.float32),
        'gcn_pred': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], label_num=y_train.shape[1],  logging=True)

    saver = tf.train.Saver()


    # Init variables
    sess.run(tf.global_variables_initializer())

    print("hushu_ms_academic_phy_teach_start2")
    checkpt_file = 'data/mod_new_teach_split_{}_{}_{}.ckpt'.format(FLAGS.dataset, FLAGS.model, k)
    cost_val = []
    acc_val = []
    val_loss_min = np.inf
    patience_step = 0

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict_un_teacher(features, support, y_train, train_mask, epoch, gcn_pred[k], placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, epoch, gcn_pred[k], placeholders)
        cost_val.append(cost)
        acc_val.append(acc)

        # Print results
        # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
        #       "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
        #       "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        # if epoch > FLAGS.early_stopping and cost_val[-1] < np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        #     print("Early stopping...")
        #     break

        # early stop for KDD
        # if len(acc_val) > 1:
        #     if acc_val[-1] >= np.max(acc_val[-(len(acc_val)):-1]):
        #         saver.save(sess, checkpt_file)
        if cost <= val_loss_min:
            val_loss_min = min(cost, val_loss_min)
            patience_step = 0
            saver.save(sess, checkpt_file)
        else:
            patience_step += 1
        if patience_step >= FLAGS.early_stopping:
            print("Early stopping...")
            break


    print("epoch: ", k)
    print("Optimization Finished!")

    # Testing
    saver.restore(sess, checkpt_file)
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, epoch, gcn_pred[k], placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    random_test.append(test_acc)
    #  MC-dropout to sample parameter, we choose 50 samples and take average
    Baye_result = []
    for p in range(100):
        feed_dict = construct_feed_dict_un_teacher(features, support, y_test, test_mask,  epoch, gcn_pred[k], placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        outs = sess.run([model.loss, model.outputs], feed_dict=feed_dict)
        Baye_result.append(outs[1])
    Bayesian_result_predict.append(Baye_result)

    Baye_acc = masked_accuracy_numpy(np.mean(Baye_result, axis=0), y_test, test_mask)
    print("Baye accuracy=", "{:.5f}".format(Baye_acc))
    Bayesian_result.append(Baye_acc)
np.save("/network/rit/lab/ceashpc/sharedata_shu/gcn_{}_{}_Baye_random_split_new.npy".format(FLAGS.dataset, FLAGS.model),Bayesian_result_predict)
print("Random accuracy=", "{:.5f}".format(np.mean(random_test)), "Random std=", "{:.5f}".format(np.std(random_test)))
print("Bayesian accuracy=", "{:.5f}".format(np.mean(Bayesian_result)), "Bayesian std=", "{:.5f}".format(np.std(Bayesian_result)))
print(random_test)
print(Bayesian_result)