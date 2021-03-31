
import pdb
import tensorflow as tf
import precog.interface as interface
import precog.utils.tensor_util as tensoru
import precog.utils.tfutil as tfu

class BinaryMaskProxy(interface.ProxyDistribution):
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def log_prob(self, samples, *args, **kwargs):
        # Treat the first channel in the features as a cost map.
        # cost_map = samples.phi.overhead_features[..., 0]
        cost_map = samples.phi.overhead_features[..., 1]
        # Reshape the rollouts for interpolation.
        rollout_reshape = tf.reshape(samples.rollout.S_grid_frame, (samples.phi_metadata.B, -1, 2))
        # Interpolate the costs.
        costs_interp = tfu.interpolate_bilinear(grid=cost_map[..., None], query_points=rollout_reshape, indexing='ij')[..., 0]
        # Reshape the costs and negate.
        rewards = -1 * tf.reshape(costs_interp, samples.rollout.batch_shape + samples.rollout.event_shape[:-1])
        # Compute sums across the trajectories (for all agents across all timesteps!)
        log_prob = tensoru.repeat_reduce_sum(rewards, n=len(samples.rollout.event_shape[:-1]), axis=-1)
        # Compute mean across samples.
        mean_log_prob = tf.reduce_mean(log_prob, axis=-1)
        return mean_log_prob
