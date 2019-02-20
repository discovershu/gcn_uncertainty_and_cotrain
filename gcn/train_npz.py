from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np

from gcn.utils import *
from gcn.models import GCN, MLP
from Load_npz_old import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)



# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'ms_academic_phy', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed', 'ms_academic_cs', 'ms_academic_phy', 'amazon_electronics_computers', 'amazon_electronics_photo'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.8, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_nell(FLAGS.dataset)
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_nell_data()
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_npz_data(FLAGS.dataset, seed)

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
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, tf.nn.softmax(model.outputs)], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]

saver = tf.train.Saver()
random_test = []
random_test_predict = []
for k in range(1):
    # Init variables
    sess.run(tf.global_variables_initializer())
    print("hushu_ms_academic_cs_start5")
    checkpt_file = 'data/mod_{}_{}.ckpt'.format(FLAGS.dataset, k)
    cost_val = []
    acc_val = []
    val_loss_min = np.inf
    patience_step = 0

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs], feed_dict=feed_dict)

        # Validation
        cost, acc, duration, predict = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)
        acc_val.append(acc)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        # originial early stop
        # if len(acc_val)>1:
        #     if acc_val[-1]> np.max(acc_val[-(len(acc_val)):-1]):
        #         saver.save(sess, checkpt_file)
        # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        #     print("Early stopping...")
        #     break

        # early stop for nell
        # if len(acc_val)>1:
        #     if acc_val[-1]> np.max(acc_val[-(len(acc_val)):-1]):
        #         saver.save(sess, checkpt_file)
        # if epoch > FLAGS.early_stopping and acc_val[-1] < np.mean(acc_val[-(10+1):-1]):
        #     print("Early stopping...")
        #     break

        # early stop for KDD
        if len(acc_val)>1:
            if acc_val[-1]> np.max(acc_val[-(len(acc_val)):-1]):
                saver.save(sess, checkpt_file)
        if cost <= val_loss_min:
            val_loss_min = min(cost, val_loss_min)
            patience_step = 0
        else:
            patience_step += 1
        if patience_step >= FLAGS.early_stopping:
            print("Early stopping...")
            break


    print("Optimization Finished!")

    # Testing
    saver.restore(sess, checkpt_file)
    test_cost, test_acc, test_duration, test_predict = evaluate(features, support, y_test, test_mask, placeholders)
    random_test.append(test_acc)
    random_test_predict.append(test_predict)
    print("k=", k)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
# np.save("/network/rit/lab/ceashpc/sharedata_shu/gcn_{}_{}.npy".format(FLAGS.dataset, 10), random_test_predict)
print("random_test =", random_test)
print("Random accuracy=", "{:.5f}".format(np.mean(random_test)), "Random std=", "{:.5f}".format(np.std(random_test)))
