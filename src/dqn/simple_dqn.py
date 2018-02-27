from src.dqn.estimator import Estimator
import os
import tensorflow as tf
import numpy as np


class SimpleDQN:

    def __init__(self, d_in, d_out, h, gamma):
        tf.reset_default_graph()
        self.exp_dir = os.path.abspath('./experiments/')
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.q_estimator = Estimator(d_in, d_out, h, "q-network", self.exp_dir)
        self.target_estimator = Estimator(d_in, d_out, h, "target-q-network")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.d_in = d_in
        self.d_out = d_out
        self.gamma = gamma

    def train(self, sample):
        state, action, reward, next_state, over = zip(*sample)
        d1 = np.shape(state)[0]
        d2 = np.shape(state)[2]
        state = np.reshape(state, (d1, d2))
        next_state = np.reshape(next_state, (d1, d2))
        q_values_next = self.target_estimator.predict(self.sess, next_state)
        y_target = reward + np.invert(over).astype(np.float32) * self.gamma *\
                            np.amax(q_values_next, axis=1)
        loss, td_error = self.q_estimator.update(self.sess, state, action,
                                                 y_target)
        return loss, td_error

    def update_target_q_network(self):
        estimator1 = self.q_estimator
        estimator2 = self.target_estimator

        e1_params = [t for t in tf.trainable_variables() if
                     t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if
                     t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)
        update_ops = []

        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        self.sess.run(update_ops)

    def get_best_action(self, state):
        y = self.q_estimator.predict(self.sess, state)
        return np.argmax(y)
