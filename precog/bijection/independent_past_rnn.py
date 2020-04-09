
import functools
import itertools
import logging
import operator
import pdb
import tensorflow as tf

import precog.interface as interface
import precog.bijection.esp_bijection as esp_bijection
import precog.utils.tensor_util as tensoru
import precog.utils.log_util as logu
import precog.utils.class_util as classu

log = logging.getLogger(__file__)

class IndependentPastRNNBijection(esp_bijection.ESPJointTrajectoryBijectionMixin, interface.ESPJointTrajectoryBijection):
    @logu.log_wrapi()
    def __init__(self, A):
        self.past_rnn = tf.contrib.rnn.GRUCell(num_units=32)
        self.past_encodings = None
        self.rnn = tf.contrib.rnn.GRUCell(num_units=32)
        self.mlp = [tf.layers.Dense(32, activation=tf.nn.relu), tf.layers.Dense(6)]
        self._A = A
        
    @property
    def A(self):
        return self._A

    @property
    def variables(self):
        return self.rnn.variables + tensoru.layer_variables(self.mlp) + self.past_rnn.variables

    def _prepare(self, batch_shape, phi):
        assert(isinstance(batch_shape, tuple))
        self.batch_shape = batch_shape
        self.batch_size = functools.reduce(operator.mul, self.batch_shape)
        self.rnn_states = [self.rnn.zero_state(self.batch_size, dtype=tf.float64) for _ in range(self.A)]
        # Create encodings once.
        if self.past_encodings is None:
            self.past_encodings = [
                tf.nn.dynamic_rnn(cell=self.past_rnn, inputs=phi.S_past_car_frames[:, a], dtype=tf.float64)[1] for a in range(self.A)]
        if len(batch_shape) > 1:
            tilex = functools.reduce(operator.mul, batch_shape[1:])
            self.past_encodings_batch = [tf.tile(e, (tilex, 1)) for e in self.past_encodings]
        else:
            self.past_encodings_batch = self.past_encodings
        self.t = 0
    
    def __repr__(self):
        return self.__class__.__name__ + "()"

    def step_generate(self, S_history, phi, *args, **kwargs):
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
            # (B, F)
            rnn_feats = tf.concat((lastlast_agent_position, last_agent_position, self.past_encodings_batch[a]), axis=-1)
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
