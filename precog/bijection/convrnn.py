
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
import precog.utils.tfutil as tfu

log = logging.getLogger(__file__)

class ConvRNN(esp_bijection.ESPJointTrajectoryBijectionMixin, interface.ESPJointTrajectoryBijection):
    @logu.log_wrapi()
    def __init__(self, A, kernel_size, conv_filters, n_conv_layers, past_gru_units, future_gru_units, mlp_units):
        self.past_rnn = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=past_gru_units)
        self.past_encodings = None
        self.feature_map = None
        self.rnn = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=future_gru_units)
        self.F = 8
        self.convnet = [
            tf.compat.v1.layers.Conv2D(filters=conv_filters, kernel_size=kernel_size, activation=tf.nn.relu, padding='same') for i in range(n_conv_layers)]
        self.convnet.append(tf.compat.v1.layers.Conv2D(filters=self.F, kernel_size=kernel_size, activation=tf.nn.relu, padding='same'))
        self.mlp = [tf.compat.v1.layers.Dense(mlp_units, activation=tf.nn.relu), tf.layers.Dense(6)]
        self._A = A
        
    @property
    def A(self):
        return self._A

    @property
    def variables(self):
        return self.rnn.variables + tensoru.layer_variables(self.mlp) + self.past_rnn.variables + tensoru.layer_variables(self.convnet)

    def _prepare(self, batch_shape, phi):
        assert(isinstance(batch_shape, tuple))
        # (B, ...), e.g. (B,); (B,K), etc.
        self.batch_shape = batch_shape
        self.batch_size = functools.reduce(operator.mul, self.batch_shape)
        self.rnn_states = [self.rnn.zero_state(self.batch_size, dtype=tf.float64) for _ in range(self.A)]

        # Determine map feature shape.
        if len(self.batch_shape) == 1: self.map_feat_shape = self.batch_shape + (1, 2)
        elif len(self.batch_shape) == 2: self.map_feat_shape = self.batch_shape + (2,)
        else: raise ValueError
            
        # Create past encodings once.
        if self.past_encodings is None:
            self.past_encodings = [
                tf.nn.dynamic_rnn(cell=self.past_rnn, inputs=phi.S_past_car_frames[:, a], dtype=tf.float64)[1] for a in range(self.A)]
        # Create feature map once.
        if self.feature_map is None:
            self.feature_map = tensoru.convnet(self.convnet, phi.overhead_features)

        # Tile the past encodings for the current batch_shape
        if len(batch_shape) > 1:
            tilex = functools.reduce(operator.mul, batch_shape[1:])
            self.past_encodings_batch = [tf.tile(e, (tilex, 1)) for e in self.past_encodings]
        else:
            self.past_encodings_batch = self.past_encodings
        self.t = 0
    
    def __repr__(self):
        return self.__class__.__name__ + "()"

    def step_generate(self, S_history, phi, *args, **kwargs):
        """

        :param S_history: Rollout history, must be in __CAR FRAMES__.
        :param phi: ESPPhi context.
        :returns: 
        :rtype: 

        """
        ms = []
        sigmas = []
        sigels = []

        # batch_shape + (A, D)
        last_agent_positions_grid = phi.local2grid.apply(S_history[-1][..., None, :])[..., 0, :]
        # (B, batch_shape[1:]*A, d)
        last_agent_positions_grid_r = tf.reshape(last_agent_positions_grid, (self.batch_shape[0], -1, 2))
        # (B, batch_shape[1:]*A, F)
        map_feat = tfu.interpolate_bilinear(self.feature_map, last_agent_positions_grid_r, indexing='ij')
        # (BSize, A, F)
        map_feats = tf.reshape(map_feat, (self.batch_size, self.A, self.F))
        # len(A) list of [batch_shape + (F,)] arrays
        per_agent_map_feats = tf.unstack(map_feats, axis=-2)
        
        for a in range(self.A):
            last_agent_position_f = S_history[-1][..., a, :]
            # (BSize, 2)
            last_agent_position = tf.reshape(last_agent_position_f, (self.batch_size, 2))
            lastlast_agent_position = S_history[-2][..., a, :]
            # (BSize, 2)
            lastlast_agent_position = tf.reshape(lastlast_agent_position, (self.batch_size, 2))
            # (BSize, G)
            rnn_feats = tf.concat((lastlast_agent_position, last_agent_position, self.past_encodings_batch[a], per_agent_map_feats[a]), axis=-1)
            # (BSize, future_gru_units)
            output, rnn_state = self.rnn(inputs=rnn_feats, state=self.rnn_states[a])
            self.rnn_states[a] = rnn_state
            # (BSize, 6)
            predictions = tensoru.mlp(self.mlp, output)
            # (Bsize, 2)
            m_ta = predictions[..., :2]
            # (Bsize, 2, 2). Make square matrices.
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
