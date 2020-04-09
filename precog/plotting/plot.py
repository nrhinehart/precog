
import functools
import logging
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pdb
import skimage.transform
import tensorflow as tf

import precog.interface as interface

import precog.utils.log_util as logu
import precog.utils.mpl_util as mplu
import precog.utils.tensor_util as tensoru
import precog.ext.nuscenes.nuscenes as nuscenes

log = logging.getLogger(__file__)

COLORS = """#377eb8
#ff7f00
#4daf4a
#984ea3
#ffd54f""".split('\n')

def astype(x, dtype):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif hasattr(x, 'numpy'):
        return astype(x.numpy(), dtype)
    else:
        raise ValueError("Unrecognized type")

def magic_reset_axis(ax):
    ax.cla()
    ax.set_aspect('equal')
    # ax.set_axis_off()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

def magic_reset(fig, ax):
    magic_reset_axis(ax)
    fig.subplots_adjust()

def cmap_discretize(cmap, N):
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1)]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

def plot_figure(key, fig, partial_write_np_image_to_tb=None):
    """Renders the figure into tensorboard. If given a function of image arrays, it'll use that after rasterizing the figure

    :param key: str tensorboard key, only needed if not using partial_write_np_image_to_tb
    :param fig: matplotlib figure
    :param partial_write_np_image_to_tb: output of bind_write_np_image_to_tb. Needs to be unique for each figure plotted with plot_figure.
    :returns: 
    :rtype: 

    """
    # Rasterize the figure
    arr = mplu.fig2rgb_array(fig)
    if partial_write_np_image_to_tb is not None: return partial_write_np_image_to_tb(arr)
    else: return tf.contrib.summary.image(key, arr)

def write_np_image_to_tb(arr, sess, image_summary, image_placeholder, writer, global_step):
    """Add the image summary to the event log

    :param arr: ndarray
    :param sess: tf.Session
    :param image_summary: summary node to evaluate
    :param image_placeholder: input placeholder of the image
    :param writer: tf.Writer
    :param global_step: global step at which to register this image
    :returns: 
    :rtype: 

    """
    image_summary_pb, gs = sess.run([image_summary, global_step], feed_dict={image_placeholder: arr})
    writer.add_summary(image_summary_pb, gs)

def bind_write_np_image_to_tb(sess, writer, global_step, C=4, key='sampled_minibatch'):
    """Create a function to plot a numpy array into tensorboard in graph mode

    :param sess: tf.Session
    :param writer: tf.summary.Writer
    :param global_step: global_step
    :returns: function of a single argument(np.ndarray) that plots to tensorboard
    :rtype: 

    """
    
    image_placeholder = tf.compat.v1.placeholder(shape=(None, None, None, C), dtype=tf.float32, name=key + '_image_ph')
    im_summary = tf.compat.v1.summary.image(key, image_placeholder)
    return functools.partial(write_np_image_to_tb,
                             sess=sess,
                             image_summary=im_summary,
                             image_placeholder=image_placeholder,
                             writer=writer,
                             global_step=global_step)
    
def plot_image(key, im, cmap=None):
    fig, ax = plt.subplots(figsize=(3,3))
    obj = ax.imshow(im, cmap=cmap)
    fig.colorbar(obj)
    plot_figure(key, fig)

def plot_trajectory(key,
                    traj,
                    fig=None,
                    ax=None,
                    colors='greens',
                    alpha=None,
                    zorder=1,
                    marker='o',
                    facecolor='none',
                    markeredgewidth=None,
                    markersize=None,
                    markerfacecolor='none',
                    markeredgecolor=None,
                    linewidth=1,
                    connect=False,
                    inflate=False,
                    color=None,
                    label=None,
                    render=False,
                    axis=None):

    if fig is None:
        assert(False)
        assert(ax is None)
        fig, ax = plt.subplots(figsize=(3,3))

    ax.plot(*traj.T,
            marker=marker,
            linewidth=linewidth,
            color=color,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            alpha=alpha,
            zorder=zorder,
            label=label)

    if axis is not None:
        ax.axis(axis)
    
    if render:
        plot_figure(key, fig)
        plt.close('all')
        return None, None
    else:
        return fig, ax
    
def plot_joint_trajectory(joint_traj, key='joint_trajectories', fig=None, ax=None, render=True, limit=100, **kwargs):
    """

    :param joint_traj: K x A x T x d
    :param key: 
    :param fig: 
    :param ax: 
    :param render: 
    :param kwargs: 
    :returns: 
    :rtype: 

    """
    
    assert(tensoru.rank(joint_traj) == 4)
    A = tensoru.size(joint_traj, 1)
    for a in range(A):
        render_a = (a == A - 1) and render
        color = kwargs.get('color', COLORS[a])
        if isinstance(joint_traj, tf.Tensor):
            single_traj = joint_traj[:, a].numpy()
        else:
            single_traj = joint_traj[:, a]
        kwargs.pop('color', None)
        fig, ax = plot_trajectory(
            key, single_traj, color=color, render=render_a, fig=fig, ax=ax, axis=[0, limit, limit, 0], **kwargs)
        assert(fig is not None)        
    return fig, ax

def plot_single_sampled_output(sampled_output, batch_index=0, fig=None, ax=None, render=True):
    S = sampled_output.rollout.S_grid_frame
    S_past = sampled_output.phi.S_past_grid_frame
    limit = tensoru.size(sampled_output.phi.overhead_features, 1)
    # Plot future.
    fig, ax = plot_joint_trajectory(S[batch_index], key='sampled_trajectories', render=False, marker='o', zorder=1, alpha=.4, fig=fig, ax=ax, limit=limit)
    return fig, ax

def plot_whiskers(rollout, batch_index=0, k_slice=slice(0,1), a_slice=slice(0,5), period=5, fig=None, ax=None, render=True):
    A, T, d = rollout.event_shape
    whiskers = [_['whiskers_grid'][batch_index, k_slice, a_slice] for _ in rollout.metadata_list[::period]]
    frames = [_['local2grid'] for _ in rollout.metadata_list]
    origins = tf.stack([_.t[batch_index, k_slice, a_slice] for _ in frames], axis=-2)
    limit = tensoru.size(rollout.phi.overhead_features, 1)

    #plot_joint_trajectory(whiskers[0], key='whiskers0', render=render, fig=fig, ax=ax, marker='d', zorder=2, alpha=0.5, limit=limit)

    plot_joint_trajectory(origins, key='origins', render=render,
                          fig=fig, ax=ax, marker='+', zorder=2, alpha=0.5, limit=limit)
    
#    plot_joint_trajectory(whiskers[-1], key='whiskers1', render=render, fig=fig, ax=ax, marker='d', zorder=2, alpha=0.5, limit=limit)
    return fig, ax

def plot_bev(sampled_output, batch_index, ax, onechannel=True, fmt='carla', allchannel=False, channel_idx=0):
    if onechannel:
        bev = sampled_output.phi.overhead_features[batch_index, ..., channel_idx]
        return ax.imshow(astype((bev*255), np.uint8), cmap='Greys', origin='upper')
    elif allchannel:
        bev = sampled_output.phi.overhead_features[batch_index]
        return ax.imshow(astype((bev*255), np.uint8), origin='upper')
    else:
        if fmt == 'nuscenes':
            below_slice = slice(0,2)
            above_slice = slice(2,100)
        else:
            below_slice = slice(2,3)
            above_slice = slice(1,2)
        bev0 = astype(sampled_output.phi.overhead_features[batch_index, ..., below_slice], np.float64).sum(axis=-1)
        # Indicates below
        grey_pixels = np.where(bev0 > 0.01)        

        bev1 = astype(sampled_output.phi.overhead_features[batch_index, ..., above_slice], np.float64).sum(axis=-1)
        red_pixels = np.where(bev1 > 0.01)
        
        image = 255 * np.ones(bev0.shape[:2] + (3,), dtype=np.uint8)
        # Grey for below
        image[grey_pixels] = [153, 153, 153]
        # Red for above
        image[red_pixels] = [228, 27, 28]
        return ax.imshow(image, origin='upper')

def plot_expert(expert, batch_index, fig, ax, limit, render=False, **kwargs):
    """

    :param expert: ESPExperts, numpy'd
    :param batch_index: 
    :param fig: 
    :param ax: 
    :param limit: 
    :param render: 
    :returns: 
    :rtype: 

    """
    
    return plot_joint_trajectory(expert.S_future_grid_frame[batch_index][None],
                                 key='expert',
                                 render=render,
                                 fig=fig,
                                 ax=ax,
                                 marker='s',
                                 zorder=3,
                                 alpha=0.5,
                                 limit=limit,
                                 **kwargs)

def load_nuscenes(version='v1.0-mini', dataroot='/home/nrhinehart/data/datasets/nuscenes_raw/', verbose=True, nusc=[None]):
    if nusc[0] is None: nusc[0] = nuscenes.NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    return nusc[0]

def plot_nuscenes_LIDAR(sampled_output, b, ax, height_colorize=True, n_sweeps=10, subsample=20):
    from precog.ext.nuscenes.utils.data_classes import LidarPointCloud
    sample_token = sampled_output.phi_metadata.metadata_list.to_dict()['sample_token'][b]
    ref_chan = 'LIDAR_TOP'
    cmap = cmap_discretize(cm.Set1_r, 2)
    
    nusc = load_nuscenes()

    # Get aggregated point cloud in LIDAR frame.
    sample_rec = nusc.get('sample', sample_token)
    LIDAR_sample_rec = nusc.get('sample_data', sample_rec['data'][ref_chan])
    LIDAR_token = LIDAR_sample_rec['token']

    log.debug("Loading nuscenes LIDAR {}".format(b))
    pc, times = LidarPointCloud.from_file_multisweep(
        nusc, sample_rec=None, chan=ref_chan, ref_chan=ref_chan, sample_data_token=LIDAR_token, nsweeps=n_sweeps)

    points = pc.points[:, ::subsample]
    
    # dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    # colors = np.minimum(1, dists/self.graph_input.consts.H_bev/np.sqrt(2))
    # ppm = self.graph_input.consts.ppm

    # TODO there's a small offset due to vehicle center vs. Lidar center difference.
    points[0] *= -1
    # Transform the points according to a special / hack transform.
    # points_scaled = LidarTform[:2, :2] @ points[:2]*ppm + self.graph_input.consts.H_bev / 2
    points_scaled = points[:2]
    w2g = sampled_output.phi.world2grid.to_numpy()
    # pdb.set_trace()
    points_grid = w2g.apply(points_scaled.T).T
    # points_scaled = points[:2]*ppm + self.graph_input.consts.H_bev / 2
    points_mask = np.logical_and(points_grid < sampled_output.phi_metadata.H, points_grid >= 0).all(axis=0)
    heights = points[2, points_mask]
    hmin, hmax = -6, 4
    heights[heights > hmax] = hmax
    heights[heights < hmin] = hmin
    hsort = np.argsort(heights)

    log.debug("Plotting nuscenes points {}".format(b))
    obj = ax.scatter(points_grid[1, points_mask][hsort],
                     points_grid[0, points_mask][hsort],
                     c=heights[hsort],
                     s=0.1,
                     alpha=0.5,
                     cmap=cmap)
    return obj

def get_figure(fig=None, axes=None, nrows=5, ncols=2, figsize=(8,20)):
    if fig is None:
        assert(axes is None)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes

@logu.log_wrapd()
def plot_sampled_minibatch(sampled_output,
                           experts=None,
                           figsize=(8,20),
                           partial_write_np_image_to_tb=None,
                           without_samples=False,
                           plot_bev_kwargs={},
                           tensorstr='',
                           fig=None,
                           axes=None,
                           eager=False):
    fig, axes = get_figure(fig=fig, axes=axes, figsize=figsize)
    B = sampled_output.phi_metadata.B
    A = sampled_output.rollout.event_shape[0]
    limit = tensoru.size(sampled_output.phi.overhead_features, 1)
    # Make the expert color different from the model if there's only one agent.
    expert_kwargs = {'color': COLORS[1]} if A == 1 else {}
    for b in range(min(B,10)):
        ax = axes.ravel()[b]
        magic_reset_axis(ax)
        if not without_samples:
            plot_single_sampled_output(sampled_output, batch_index=b, render=False, fig=fig, ax=ax)

        plot_past(sampled_output.phi.S_past_grid_frame, b=b, fig=fig, ax=ax, limit=limit)
        plot_rollout(sampled_output.rollout, fig=fig, ax=ax, b=b)
        plot_bev(sampled_output, batch_index=b, ax=ax, **plot_bev_kwargs)
        if experts is not None:
            plot_expert(
                experts,
                batch_index=b,
                fig=fig,
                ax=ax,
                render=False,
                limit=tensoru.size(sampled_output.phi.overhead_features, 1),
                **expert_kwargs)
    # TODO make work with eager mode.            
    plot_figure('sampled_minibatch' + tensorstr, fig, partial_write_np_image_to_tb=partial_write_np_image_to_tb)
    if not eager: figsclose()
    return fig

def figsclose(): plt.close('all')
    
def plot_feature_map(feature_map,
                     partial_write_np_image_to_tb,
                     b=0,
                     nrows=3,
                     ncols=3,
                     fig=None,
                     axes=None):
    if feature_map is None: return
    figsize = (2 * ncols, 2 * nrows)
    fig, axes = get_figure(fig=fig, axes=axes, figsize=figsize, nrows=nrows, ncols=ncols)
    B, H, W, C = tensoru.shape(feature_map)
    scalar_maps = feature_map[b]
    for c in range(C):
        ax = axes.ravel()[c]
        scalar_map = scalar_maps[..., c]
        ax.imshow(scalar_map, origin='upper')
    for ax in axes.ravel():
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
    plot_figure('CNN_features_item_{}'.format(b), fig, partial_write_np_image_to_tb=partial_write_np_image_to_tb)
    
def plot_past(S_past, b, fig, ax, limit, alpha=0.5):
    # Plot past.
    plot_joint_trajectory(S_past[b][None], key='sampled_trajectories', render=False, fig=fig, ax=ax, marker='d', zorder=2, alpha=alpha, limit=limit)
    # Plot origin.
    plot_joint_trajectory(S_past[b][None][..., -1, :][..., None, :],
                          key='sampled_trajectories', render=False,
                          fig=fig, ax=ax, marker='d', zorder=10, alpha=alpha, color='r', limit=limit)
    

def plot_rollout(rollout, b=None, fig=None, ax=None):
    assert(b is not None)
    limit = tensoru.size(rollout.phi.overhead_features, 1)
    plot_joint_trajectory(joint_traj=rollout.S_grid_frame[b],
                          key='rollout_future',
                          render=False,
                          fig=fig,
                          ax=ax,
                          marker='o',
                          zorder=3,
                          alpha=0.5,
                          limit=limit)

@logu.log_wrapd()
def plot_sample(sampled_output, expert=None, b=0, figsize=(4,4), partial_write_np_image_to_tb=None, bev_kwargs={}):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    magic_reset_axis(ax)
    plot_single_sampled_output(sampled_output, batch_index=b, render=False, fig=fig, ax=ax)
    plot_bev(sampled_output, batch_index=b, ax=ax, **bev_kwargs)
    A = tensoru.size(sampled_output.rollout.S_car_frames, 2)
    expert_kwargs = {'color': COLORS[1]} if A == 1 else {}
    limit = tensoru.size(sampled_output.phi.overhead_features, 1)
    plot_past(sampled_output.phi.S_past_grid_frame, b=b, fig=fig, ax=ax, limit=limit)
    if expert is not None:
        plot_expert(expert, batch_index=b, fig=fig, ax=ax, render=False, limit=tensoru.size(sampled_output.phi.overhead_features, 1), **expert_kwargs)
    res = plot_figure('sampled_minibatch', fig, partial_write_np_image_to_tb=partial_write_np_image_to_tb)
    plt.close('all')
    return res

def plot_coordinate_frames(coordinate_frame_list, b=0, fig=None, axes=None, figsize=(8,20)):
    """Coordinate frames should map local coordinates to grid coordinates"""
    fig, axes = get_figure(fig=fig, axes=axes, figsize=figsize)
    # TODO Debugging hacks hardcoded.
    points_local = tf.convert_to_tensor(np.tile(np.array((1,0), dtype=np.float64)[None, None, None], (10, 12, 5, 1)))
    bases_grid = np.concatenate([_.apply(points_local, points_ein='bkaj')[b].numpy().reshape((-1,2)) for _ in coordinate_frame_list],0)
    origins = np.concatenate([_.t[b].numpy().reshape((-1,2)) for _ in coordinate_frame_list], 0)
    axes.ravel()[b].plot(origins[:,0], origins[:,1], marker='x')
    axes.ravel()[b].plot(bases_grid[:,0], bases_grid[:,1], marker='o')
    return fig, axes
