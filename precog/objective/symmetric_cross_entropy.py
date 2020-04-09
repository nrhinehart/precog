
import logging
import pdb

import precog.interface as interface
import precog.utils.np_util as npu
import precog.utils.class_util as classu

log = logging.getLogger(__file__)

class SymmetricCrossEntropy(interface.ESPObjective):
    @property
    def requires_samples(self): return True
    
    @classu.member_initialize
    def __init__(self, beta=0.0001):
        pass
        
    def call(self, density_distribution, density_distribution_samples, target_distribution_proxy, minibatch, data_epsilon):
        """

        :param density_distribution: 
        :param minibatch: 
        :returns: ESPObjectiveReturn(minimization criterion, H(p,q), ehat, rollout)
        :rtype: 

        """
        assert(str(target_distribution_proxy) != 'EmptyProxy()')
        log_prob, roll = density_distribution.log_prob(S_future=minibatch.experts.S_future_car_frames, phi=minibatch.phi)
        forward_cross_entropy = -1 * log_prob
        reverse_cross_entropy = -1 * target_distribution_proxy.log_prob(density_distribution_samples)
        H_lb = npu.entropy_lower_bound(k=roll.dimensionality, stddev=data_epsilon)
        ehat = (forward_cross_entropy - H_lb) / roll.dimensionality
        min_criterion = forward_cross_entropy + self.beta * reverse_cross_entropy
        return interface.ESPObjectiveReturn(
            min_criterion=min_criterion,
            forward_cross_entropy=forward_cross_entropy,
            ehat=ehat,
            rollout=roll,
            reverse_cross_entropy=reverse_cross_entropy)
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + "()"
