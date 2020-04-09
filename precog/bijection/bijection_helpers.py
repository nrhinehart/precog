
import pdb
import numpy as np
import tensorflow as tf

import precog.utils.tfutil as tfu
import precog.utils.tensor_util as tensoru

def get_map_feats(feature_map, batch_shape, batch_size, last_agent_positions_grid, A, phi=None):
    """Interpolate the positions into the feature map.

    :param feature_map: (B, H, W, F), feature map
    :param batch_shape: first several dimensions of positions. could be inferred but we require it for sanity checks.
    :param batch_size: size of batch. could be inferred but we require it for sanity checks.
    :param last_agent_positions_grid: batch_shape + (A, D)
    :param A: number of agents. could be inferred but we require it for sanity checks.
    :param F: feature map dimension. could be inferred but we require it for sanity checks.
    :param phi: DEPRECATED
    :returns: 
    :rtype: 

    """
    assert(tensoru.rank(feature_map) == 4)
    assert(tensoru.rank(last_agent_positions_grid) >= 3)
    F = tensoru.size(feature_map, -1)

    # (B, batch_shape[1:]*A, d)
    last_agent_positions_grid_r_xy = tf.reshape(last_agent_positions_grid, (batch_shape[0], -1, 2))
    last_agent_positions_grid_r_ij = last_agent_positions_grid_r_xy[..., ::-1]
    # (B, batch_shape[1:]*A, F)
    # N.B. the indexing order! Our points are stored in xy-format, with x corresponding to the W dimension of the feature grid.
    map_feat = tfu.interpolate_bilinear(feature_map, last_agent_positions_grid_r_ij, indexing='ij')
    # (BSize, A, F)
    map_feats = tf.reshape(map_feat, (batch_size, A, F))
    return map_feats

def get_social_spatial_differences(phi, batch_shape, batch_size, last_agent_positions_world, A):
    # Last agent positions in world coords.
    # Replicate along the last-to-last axis.
    last_agent_positions_world_rep = tf.keras.backend.repeat_elements(last_agent_positions_world[..., None,:], rep=A, axis=-2)
    # Transpose the [... A, A, 2] matrix so that the ensuing similarity transform operates on the other agents.
    last_agent_positions_world_rep = tensoru.swap_axes(last_agent_positions_world_rep, -3, -2)
    # Outer-product of similarity transforms. Relative distances to the other agents.
    # [..., i, :, :] is the relative distances of the other agents to agent i in agent i's coordinate frame.
    last_agent_positions_local_outer = phi.world2local.apply(last_agent_positions_world_rep)
    # Prepare position feats for mlp.
    last_agent_position_feats = tf.reshape(last_agent_positions_local_outer, batch_shape + (A, -1))
    return last_agent_position_feats

def get_whisker_map_feats(feature_map, batch_shape, batch_size, cars2grid, template_cars, A, phi=None):
    """Interpolate the positions into the feature map.

    :param feature_map: (B, H, W, F), feature map
    :param batch_shape: first several dimensions of positions. could be inferred but we require it for sanity checks.
    :param batch_size: size of batch. could be inferred but we require it for sanity checks.
    :param last_agent_positions_cars: batch_shape + (A, D)
    :param cars2grid: SimilarityTransform from car frames to grid
    :param template: (N, D) template of positions in local frame at which to interpolate
    :param A: number of agents. could be inferred but we require it for sanity checks.
    :param F: feature map dimension. could be inferred but we require it for sanity checks.
    :param phi: DEPRECATED
    :returns: 
    :rtype: 

    """
    
    assert(tensoru.rank(feature_map) == 4)
    F = tensoru.size(feature_map, -1)

    if len(batch_shape) == 2:
        points_ein = 'bkaNj'
        template_cars = template_cars[None]
    elif len(batch_shape) == 1:
        points_ein = 'baNj'
    else:
        raise ValueError

    n_whiskers = tensoru.size(template_cars,-2)
    whiskers_grid = cars2grid.apply(template_cars, points_ein=points_ein)
    whiskers_grid_r_xy = tf.reshape(whiskers_grid, (batch_shape[0], -1, 2))
    # (B, batch_shape[1:]*A, F)
    # N.B. the indexing order! Our points are stored in xy-format, with x corresponding to the W dimension of the feature grid.
    map_feat = tfu.interpolate_bilinear(feature_map, whiskers_grid_r_xy, indexing='xy')
    # (B, ..., A, n_whiskers, F)
    map_feat_r = tf.reshape(map_feat, batch_shape + (A, n_whiskers, F))
    # (B*..., A, n_whiskers*F)
    map_feats = tf.reshape(map_feat_r, (batch_size, A, n_whiskers * F))
    return map_feats, whiskers_grid

def generate_whisker_template(
        radii=[1, 2, 4, 8, 16, 32], arclength=5*np.pi/4, n_samples=7):

    angle_offset = -arclength / 2.
    angles = np.linspace(angle_offset, arclength + angle_offset, n_samples)

    radii = np.asarray(radii)[:, None]
    x = (radii * np.cos(angles)).ravel()
    y = (radii * np.sin(angles)).ravel()
    xy = np.stack((x,y), axis=-1)
    return xy
