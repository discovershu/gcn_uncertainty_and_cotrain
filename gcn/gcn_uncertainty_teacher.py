from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP, GCN_uncertainty, GCN_uncertainty_teacher, GCN_EDL, GCN_uncertainty_teacher_cotrain
from gcn.metrics import *
from Load_npz_old import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed', 'nell.0.001'
flags.DEFINE_string('model', 'gcn_un_t', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense', 'gcn_un_t', 'gcn_edl', 'gcn_un_t_cotrain'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.') #64
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).') # 0.1
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.') #1e-5
flags.DEFINE_float('weight_teacher', 1.0, 'Weight for teacher KL.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_nell(FLAGS.dataset)
# gcn_pred = np.load("data/gat_{}_10.npy".format(FLAGS.dataset))
gcn_pred = np.load("/network/rit/lab/ceashpc/sharedata_shu/gcn_un_epstemic/gcn_{}_10000.npy".format(FLAGS.dataset))
# gcn_pred = []
# for i in range(10):
#     gcn_pred_i = np.load("/network/rit/lab/ceashpc/sharedata_shu/gat_baseline/gat_{}_{}.npy".format(FLAGS.dataset, i+1))
#     gcn_pred.append(gcn_pred_i)

print("Dataset : ", FLAGS.dataset)

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

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# Define model evaluation function
def evaluate(features, support, labels, mask, epoch, gcn_pred, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict_un_teacher(features, support, labels, mask, epoch, gcn_pred, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

random_test = []  # random result
Bayesian_result = []  # Bayesian result
for k in range(1):
    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict_un_teacher(features, support, y_train, train_mask, epoch, gcn_pred, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, epoch, gcn_pred, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        # if epoch > FLAGS.early_stopping and cost_val[-1] < np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        #     print("Early stopping...")
        #     break

    print("epoch: ", k)
    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, epoch, gcn_pred, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    random_test.append(test_acc)
    #  MC-dropout to sample parameter, we choose 50 samples and take average
    Baye_result = []
    hidden = []
    for p in range(500):
        feed_dict = construct_feed_dict_un_teacher(features, support, y_test, test_mask,  epoch, gcn_pred, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        outs = sess.run([model.loss, model.outputs, model.hidden1], feed_dict=feed_dict)
        Baye_result.append(outs[1])
        hidden.append(outs[2])
    Baye_acc = masked_accuracy_numpy(np.mean(Baye_result, axis=0), y_test, test_mask)
    print("Baye accuracy=", "{:.5f}".format(Baye_acc))
    Bayesian_result.append(Baye_acc)
np.save("/network/rit/lab/ceashpc/sharedata_shu/gcn_un_epstemic/gcn_{}_baye_{}.npy".format(FLAGS.dataset,FLAGS.epochs),Baye_result)
np.save("/network/rit/lab/ceashpc/sharedata_shu/gcn_un_epstemic/hidden_feature_{}_{}.npy".format(FLAGS.dataset,FLAGS.epochs),np.mean(hidden, axis=0))
print("Random accuracy=", "{:.5f}".format(np.mean(random_test)), "Random std=", "{:.5f}".format(np.std(random_test)))
print("Bayesian accuracy=", "{:.5f}".format(np.mean(Bayesian_result)), "Bayesian std=", "{:.5f}".format(np.std(Bayesian_result)))
print(random_test)
print(Bayesian_result)