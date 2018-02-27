import tensorflow as tf
import os


class Estimator:
    """ Class for both Q-Network and Target Network"""

    def __init__(self, d_in, d_out, h, scope="estimator", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        self.d_in = d_in
        self.d_out = d_out
        self.h = h

        # Placeholders
        self.X = None
        self.Y = None
        self.A = None
        self.predictions = None
        self.td_error = None
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self.summaries = None

        with tf.variable_scope(scope):
            self.build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".
                                           format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def build_model(self):
        """Builds the TensorFlow Computational Graph"""
        # Architecture of the neural network
        h1 = self.h  # Number of nodes in the first hidden layer
        lr = 1e-3  # Learning rate

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.d_in])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None])  # Target Y
        self.A = tf.placeholder(dtype=tf.int32, shape=[None])

        batch_size = tf.shape(self.X)[0]

        hidden1 = tf.layers.dense(self.X, h1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, h1, activation=tf.nn.relu)
        self.predictions = tf.layers.dense(hidden2, self.d_out)

        gather_indices = tf.range(0, batch_size) * self.d_out + self.A
        Y_pred = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        self.td_error = tf.abs(self.Y-Y_pred)
        self.loss = tf.losses.mean_squared_error(self.Y, Y_pred)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=tf.contrib.
                                                framework.get_global_step())

        self.summaries = tf.summary.merge(
            [tf.summary.scalar("loss", self.loss),
             tf.summary.histogram("q_values_hist", self.predictions),
             tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))]
        )

    def predict(self, sess, state):
        """Perform a forward pass to predict Q-values for the state"""
        return sess.run(self.predictions, {self.X: state})

    def update(self, sess, state, action, y_target):
        """Update the estimator"""
        feed_dict = {self.X: state, self.A: action, self.Y: y_target}
        summaries, global_step, _, loss, td_error = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(),
             self.train_op, self.loss, self.td_error], feed_dict=feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            # self.summary_writer.add_graph(sess.graph)

        return loss, td_error
