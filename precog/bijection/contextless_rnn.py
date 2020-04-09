
import functools
import logging
import operator
import pdb
import tensorflow as tf

import interface
import bijection.esp_bijection as esp_bijection
import utils.tensor_util as tensoru
import utils.log_util as logu
import utils.class_util as classu

log = logging.getLogger(__file__)

class ContextlessRNNBijection(esp_bijection.ESPJointTrajectoryBijectionMixin, interface.ESPJointTrajectoryBijection):
    @logu.log_wrapi()
    def __init__(self, A):
        self.rnn = tf.contrib.rnn.GRUCell(num_units=32)
        self.mlp = [tf.layers.Dense(32, activation=tf.nn.tanh), tf.layers.Dense(6)]
        self._A = A
        
    @property
    def A(self):
        return self._A

    @property
    def variables(self):
        return self.rnn.variables + tensoru.layer_variables(self.mlp)

    def _prepare(self, batch_shape, phi):
        assert(isinstance(batch_shape, tuple))
        self.batch_shape = batch_shape
        self.batch_size = functools.reduce(operator.mul, self.batch_shape)
        self.rnn_states = [self.rnn.zero_state(self.batch_size, dtype=tf.float64) for _ in range(self.A)]
        self.t = 0
    
    def __repr__(self):
        return self.__class__.__name__ + "()"

    def current_metadata(self):
        return {}

    def step_generate(self, S_history, *args, **kwargs):
        """Use m_t=0 and sigma_t=I. For debug purposes.

        :param S_history: 
        :returns: 
        :rtype: 

        """
        ms = []
        sigmas = []
        sigels = []
        
        for a in range(self.A):
            last_agent_position = S_history[-1][..., a, :]
            last_agent_position = tf.reshape(last_agent_position, (-1, 2))
            lastlast_agent_position = S_history[-2][..., a, :]
            lastlast_agent_position = tf.reshape(lastlast_agent_position, (-1, 2))
            # time_feat = self.t * tf.ones_like(last_agent_position)
            rnn_feats = tf.concat((lastlast_agent_position, last_agent_position), axis=-1)
            output, rnn_state = self.rnn(inputs=rnn_feats, state=self.rnn_states[a])
            self.rnn_states[a] = rnn_state
            predictions = tensoru.mlp(self.mlp, output)
            m_ta = predictions[..., :2]
            # Make square matrices.
            sigel = tf.reshape(predictions[..., 2:], (-1, 2, 2))

            # Reshape things if we have a meaningful batch size.
            if len(self.batch_shape) > 1:
                m_ta = tf.reshape(m_ta, self.batch_shape + (2,))
                sigel = tf.reshape(sigel, self.batch_shape + (2, 2))

            sigma_ta = tf.linalg.expm(sigel)

            ms.append(m_ta)
            sigels.append(sigel)
            sigmas.append(sigma_ta)

        self.t += 1
        m_t = tf.stack(ms, axis=-2)
        sigel_t = tf.stack(sigels, axis=-3)
        sigma_t = tf.stack(sigmas, axis=-3)
        return m_t, sigel_t, sigma_t
