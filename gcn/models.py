from gcn.layers import *
from gcn.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.vars2 = {}
        self.placeholders = {}

        self.layers = []
        self.layers2 = []
        self.activations = []
        self.activations2 = []
        self.kl = 0


        self.inputs = None
        self.outputs = None
        self.outputs2 = None
        self.hidden1 = None

        self.loss = 0
        self.loss2 = 0
        self.lossall = 0
        self.accuracy = 0
        self.accuracy2 = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Build sequential layer model2
        self.activations2.append(self.inputs)
        for layer in self.layers2:
            hidden = layer(self.activations2[-1])
            self.activations2.append(hidden)
        self.outputs2 = self.activations2[-1]
        self.hidden1 = self.activations[-2]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Store model variables for easy access for model2
        variables2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars2 = {var.name: var for var in variables2}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN_uncertainty(Model):
    def __init__(self, placeholders, input_dim, label_num, **kwargs):
        super(GCN_uncertainty, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.label_num = label_num
        self.annealing_step = placeholders['annealing_step']
        self.annealing_coef = tf.minimum(1.0, self.annealing_step / 50.0)
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # self.prob = self.get_alpha()
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_square_error_dirichlet(self.outputs, self.placeholders['labels'],
                                                   self.placeholders['labels_mask'])

        self.loss += self.annealing_coef * masked_kl_edl(self.outputs, self.placeholders['labels'],
                                                                self.label_num,
                                                                self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN_uncertainty_teacher(Model):
    def __init__(self, placeholders, input_dim, label_num, **kwargs):
        super(GCN_uncertainty_teacher, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.label_num = label_num
        self.annealing_step = placeholders['annealing_step']
        self.annealing_coef = tf.minimum(1.0, self.annealing_step / 200.0)
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # self.prob = self.get_alpha()
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_square_error_dirichlet(self.outputs, self.placeholders['labels'],
                                                    self.placeholders['labels_mask']) # for cora and citeseer
        # self.kl = tf.reduce_max(self.outputs)
        # self.loss += masked_square_error_dirichlet_fornell(self.outputs, self.placeholders['labels'],
        #                                            self.placeholders['labels_mask']) # for nell

        # self.loss += FLAGS.weight_teacher * masked_kl_teacher(self.outputs, self.placeholders['gcn_pred'])
        self.loss += self.annealing_coef * masked_kl_teacher(self.outputs, self.placeholders['gcn_pred']) # for cora and citeseer
        # self.loss += self.annealing_coef * masked_kl_teacher_fornell(self.outputs, self.placeholders['gcn_pred']) # for nell

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GCN_image_uncertainty_teacher(Model):
    def __init__(self, placeholders, input_dim, label_num, **kwargs):
        super(GCN_image_uncertainty_teacher, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.label_num = label_num
        self.annealing_step = placeholders['annealing_step']
        self.annealing_coef = tf.minimum(1.0, self.annealing_step / 200.0)
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # self.prob = self.get_alpha()
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        # self.loss += masked_square_error_dirichlet(self.outputs, self.placeholders['labels'],
        #                                             self.placeholders['labels_mask']) # for cora and citeseer
        # self.kl = tf.reduce_max(self.outputs)
        self.loss += masked_square_error_dirichlet_fornell(self.outputs, self.placeholders['labels'],
                                                   self.placeholders['labels_mask']) # for nell

        # self.loss += FLAGS.weight_teacher * masked_kl_teacher(self.outputs, self.placeholders['gcn_pred'])
        # self.loss += self.annealing_coef * masked_kl_teacher(self.outputs, self.placeholders['gcn_pred']) # for cora and citeseer
        self.loss += self.annealing_coef * masked_kl_teacher_forimage(self.outputs, self.placeholders['gcn_pred']) # for nell

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
        #                                     output_dim=FLAGS.hidden1,
        #                                     placeholders=self.placeholders,
        #                                     act=tf.nn.relu,
        #                                     dropout=True,
        #                                     bias=False,
        #                                     sparse_inputs=False,
        #                                     logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN_uncertainty_teacher_cotrain(Model):
    def __init__(self, placeholders, input_dim, label_num, **kwargs):
        super(GCN_uncertainty_teacher_cotrain, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.label_num = label_num
        self.annealing_step = placeholders['annealing_step']
        self.annealing_coef = tf.minimum(1.0, self.annealing_step / 200.0)
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # self.prob = self.get_alpha()
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        # self.loss+=masked_square_error_dirichlet(self.outputs, self.placeholders['labels'],
        #                               self.placeholders['labels_mask']) # for cora and citeseer
        self.loss += masked_square_error_dirichlet_fornell(self.outputs, self.placeholders['labels'],
                                                   self.placeholders['labels_mask']) # for nell


        # self.loss += FLAGS.weight_teacher * masked_kl_teacher(self.outputs, self.placeholders['gcn_pred'])
        # self.loss += self.annealing_coef * masked_kl_teacher(self.outputs, self.placeholders['gcn_pred']) # for cora and citeseer
        self.loss += self.annealing_coef * masked_kl_teacher_fornell(self.outputs, self.placeholders['gcn_pred'])  # for nell
        # self.loss += masked_kl_teacher(self.outputs, self.placeholders['gcn_pred'])

        # For loss2
        # Weight decay loss
        for var in self.layers2[0].vars2.values():
            self.loss2 += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss2 += masked_softmax_cross_entropy(self.outputs2, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'])# Cross entropy error

        # For lossall
        # self.loss = self.loss + self.loss2 + self.annealing_coef * masked_kl_cotrain(self.outputs, self.outputs2)# for cora and citeseer
        self.loss = self.loss + self.loss2 + self.annealing_coef * masked_kl_cotrain_fornell(self.outputs, self.outputs2)# for nell
        # self.loss = self.loss + self.loss2


    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

        # self.accuracy = masked_accuracy(self.outputs2, self.placeholders['labels'],
        #                                 self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

        # For model2
        self.layers2.append(GraphConvolution2(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers2.append(GraphConvolution2(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))



    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN_EDL(Model):
    def __init__(self, placeholders, input_dim, label_num, **kwargs):
        super(GCN_EDL, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.label_num = label_num
        self.annealing_step = placeholders['annealing_step']
        self.annealing_coef = tf.minimum(1.0, self.annealing_step / 10.0)
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # self.prob = self.get_alpha()
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_square_error_edl(self.outputs, self.placeholders['labels'],
                                                    self.placeholders['labels_mask'])

        self.loss += self.annealing_coef * masked_kl_edl(self.outputs, self.placeholders['labels'],
                                                               self.label_num,
                                                               self.placeholders['labels_mask'])


    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
