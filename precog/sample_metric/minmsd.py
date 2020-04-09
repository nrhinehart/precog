
from __future__ import print_function

import logging
import os
import pdb
import tensorflow as tf

import precog.interface as interface
import precog.utils.tensor_util as tensoru

log = logging.getLogger(os.path.basename(__file__))

class ESPMinMSD(interface.ESPSampleMetric):
    def call(self, model_distribution_samples, data_distribution_samples):
        S_hats = model_distribution_samples.rollout.S_world_frame
        S_star = data_distribution_samples.S_future_world_frame
        delta = S_star[:, None] - S_hats
        msd = tf.einsum('bkatd,bkatd->bk', delta, delta)
        minmsd = tf.reduce_min(msd, axis=-1)
        return minmsd

class MHat(interface.ESPSampleMetric):
    def call(self, model_distribution_samples, data_distribution_samples):
        S_hats = model_distribution_samples.rollout.S_world_frame
        S_star = data_distribution_samples.S_future_world_frame
        delta = S_star[:, None] - S_hats
        # Reshape (..., K, A, T, d) -> (..., K, ATd)
        delta_vec = tf.reshape(delta, tensoru.shape(delta)[:-3] + (-1,))
        # <z,z>
        msd = tf.einsum('...z,...z->...', delta_vec, delta_vec)
        # (..., K) -> (...,)
        minmsd = tf.reduce_min(msd, axis=-1)
        AT = float((S_star.shape[-3] * S_star.shape[-2]).value)
        mhat = tf.identity(minmsd / AT, name='mhat')
        return mhat

    def __repr__(self):
        return self.__class__.__name__ 
