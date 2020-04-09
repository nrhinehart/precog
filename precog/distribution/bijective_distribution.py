
import logging
import pdb
import tensorflow as tf
import tensorflow_probability as tfp

import precog.interface as interface

import precog.utils.class_util as classu
import precog.utils.log_util as logu

log = logging.getLogger(__file__)

class ESPBijectiveDistribution(interface.ESPDistribution):
    @logu.log_wrapi()
    @classu.member_initialize
    def __init__(self, bijection, name, dtype=tf.float64, K=6, sample_K=6, debug_logdet=False, logdet_method='trace'):
        self.base = tfp.distributions.MultivariateNormalDiag(
            loc=tf.constant(0., dtype=dtype), scale_diag=tf.constant([1., 1.], dtype=dtype))
        assert(self.logdet_method in ('trace', 'slogdet', 'manual'))

    @property
    def A(self):
        return self.bijection.A
        
    @logu.log_wrapd()
    def log_prob(self, S_future, phi):
        """Output-space log prob, using output-space input
           log q(x) = log p (finv(x)) - log | det df(f^{-1}(x))/dz |   [<-uses this form]
                    = log p (finv(x)) + log | det df^{-1}(x)/dx |      [<-doesn't use this form]

        :param S_future: in __CAR_FRAMES__
        :param phi: 
        :returns: 
        :rtype: 

        """
        # TODO record the intermediate nodes here to collect this inverse computation for inference.
        return self._log_prob_roll(self.bijection.inverse(S=S_future, phi=phi))

    def log_prob_from_Z(self, Z, phi):
        """Output-space log prob, using input-space input

        :param Z: 
        :param phi: 
        :returns: 
        :rtype: 

        """
        return self._log_prob_roll(self.bijection.forward(Z=Z, phi=phi))

    def _log_prob_roll(self, roll):
        """Computes the output-space pdf of a rollout. Doesn't require any forward or inverse computation.

        :param roll: 
        :returns: 
        :rtype: 

        """
        # --------------------------------------------
        # Compute abs log dets across T. (B, K, A, T).
        # --------------------------------------------
        if self.logdet_method == 'trace':
            # Note det(expm(A)) = e^(tr(A)) -> log det (expm(A)) = log e^(tr(A)) = tr(A)  [!!!]
            # Taking the trace should make gradients etc. more stable, assuming that the tf.linalg.expm output is a good approximation to expm(A).
            log_dets = tf.linalg.trace(roll.sigel)
        elif self.logdet_method == 'manual':
            log_dets = tf.log(tf.abs(tf.linalg.det(roll.sigma)) + 1.5e-7)
        elif self.logdet_method == 'slogdet':
            # This will do some LU decompositions, which will fail if roll.sigma is nearly singular.
            #  The documentation doesn't say that slogdet requires a PD matrix, but the documentation for logdet does say that.
            log_dets = tf.linalg.slogdet(roll.sigma)[1]
        else:
            raise ValueError("foo")

        if self.debug_logdet:
            # Take a log of the abs det after adding a bit of positive eps
            log_dets_alt = tf.log(tf.abs(tf.linalg.det(roll.sigma)) + 1.5e-7)
            with tf.control_dependencies([tf.print("Logdet mean diff:", tf.reduce_mean(log_dets - log_dets_alt))]):
                # (B, K, A). Sum across T
                log_det = tf.reduce_sum(log_dets, axis=-1)
        else:
            log_det = tf.reduce_sum(log_dets, axis=-1)
            
        # (B, K). Sum across A
        log_det = tf.reduce_sum(log_det, axis=-1)
        # Compute log probs across T. (B, K, A, T)
        log_bases = self.base.log_prob(roll.Z)
        # (B, K, A). Sum across T
        log_base = tf.reduce_sum(log_bases, axis=-1)
        # (B, K). Sum across A.
        log_base = tf.reduce_sum(log_base, axis=-1)
        # (B, K). Compute the final, per trajectory, log probs.
        log_prob = log_base - log_det
        return log_prob, roll

    def sample(self, phi, phi_metadata, T, inner=None):
        if inner is None: inner = self.sample_K
        outer = phi_metadata.B
        
        # Create the base-space samples.
        Z = tf.identity(self.base.sample(sample_shape=(outer, inner, self.A, T)), name='Z_sample')
        # Warp the base space samples forward (expensive).
        roll = self.bijection.forward(Z, phi)
        # Compute the output-space probability (cheap). 
        log_prob, _ = self._log_prob_roll(roll)
        # Record the rollout, the inputs, and the rollout's prob. This prob is important because we'll use it for DIM planning in Z-space.
        base_and_log_q = interface.ESPBaseSamplesAndLogQ(Z, log_q_samples=tf.identity(log_prob, 'log_q_samples'))
        return interface.ESPSampledOutput(rollout=roll, phi=phi, phi_metadata=phi_metadata, base_and_log_q=base_and_log_q)

    def __repr__(self):
        return self.__class__.__name__ + "(f={}, name={})".format(self.bijection, self.name)

    @property
    def variables(self):
        return self.bijection.variables
