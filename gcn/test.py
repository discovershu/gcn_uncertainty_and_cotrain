import tensorflow as tf


a = tf.ones([10, 5])
b = tf.reduce_sum(a, axis=1)
c = tf.reduce_sum(a, axis=1, keep_dims=True)
print b.shape, c.shape