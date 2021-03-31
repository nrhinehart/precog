
import logging
import numpy as np
import pdb
import tensorflow as tf

import precog.interface as interface
import precog.utils.np_util as npu
import precog.utils.tensor_util as tensoru
# import utils.class_util as classu

log = logging.getLogger(__file__)

class ForwardKL(interface.ESPObjective):
    @property
    def requires_samples(self): return False
    
    def call(self, density_distribution, density_distribution_samples, target_distribution_proxy, minibatch):
        """

        :param density_distribution: 
        :param minibatch: 
        :returns: ESPObjectiveReturn(minimization criterion, H(p,q), ehat, rollout)
        :rtype: 

        """
        assert(str(target_distribution_proxy) == 'EmptyProxy()')
        # Compute log prob (in car frames!)
        if not self.perturb:
            log.warning("Not perturbing!")
            log_prob, expert_roll = density_distribution.log_prob(S_future=minibatch.experts.S_future_car_frames, phi=minibatch.phi)
        else:
            noise = tf.random.normal(
                mean=0.,
                stddev=self.perturb_epsilon,
                shape=(self.K_perturb,) + tensoru.shape(minibatch.experts.S_future_car_frames),
                dtype=tf.float64)
            noisy_experts = tensoru.swap_axes(minibatch.experts.S_future_car_frames + noise, 0, 1)
            log_prob, expert_roll = density_distribution.log_prob(S_future=noisy_experts, phi=minibatch.phi)
        
        forward_cross_entropy = tf.identity(-1 * log_prob, 'H_pq')
        
        H_lb = npu.entropy_lower_bound(k=expert_roll.dimensionality, stddev=self.perturb_epsilon)
        ehat = tf.identity((forward_cross_entropy - H_lb) / expert_roll.dimensionality, 'ehat')
        
        return interface.ESPObjectiveReturn(
            min_criterion=forward_cross_entropy,
            forward_cross_entropy=forward_cross_entropy,
            ehat=ehat,
            rollout=expert_roll)    

    def __repr__(self):
        return self.__class__.__name__ + "()"
