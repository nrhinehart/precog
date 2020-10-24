
import atexit
import logging
import functools
import itertools
import pdb
import os

import hydra
import scipy.stats
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

import precog.utils.log_util as logu
import precog.utils.tfutil as tfutil
import precog.interface as interface
import precog.plotting.plot as plot
import precog.plotting as plotting

log = logging.getLogger(os.path.basename(__file__))    

def collect_sample(sessrun, inference):
    """

    Notes
    =====
    - ESPInference.metadata is a attrdict.AttrDict with keys B, K, A, T, and D
    - sampled_output is ESPSampledOutput and contains a member ESPRollout rollout
    - experts is ESPExperts
    - ESPSampledOutput.rollout.S_grid_frame has shape
        (B, K, A, T, D) = (1, 12, 5, 20, 2)
    - ESPExperts.S_future_grid_frame has shape
        (B, A, T, D) = (1, 5, 20, 2)

    subplots: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.subplots.html
    plot: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.axes.Axes.plot.html
    scatter plot: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.scatter.html
    setting colors: https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib
    multiple datasets in scatter: https://stackoverflow.com/questions/4270301/matplotlib-multiple-datasets-on-the-same-scatter-plot
    histogram series: https://stackoverflow.com/questions/52658364/how-to-generate-a-series-of-histograms-on-matplotlib/52659919
    embedding boxplots: https://stackoverflow.com/questions/5938459/combining-plt-plotx-y-with-plt-boxplot
    gridspec: https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/gridspec_nested.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-nested-py

    Todo
    ====
    Not sure whether expert trajectory given agent a is the same for all batches(?) b.
    """
    sampled_output = inference.sampled_output.to_numpy(sessrun)
    experts = inference.training_input.experts.to_numpy(sessrun)
    print("sampled_output.rollout.S_grid_frame.shape", sampled_output.rollout.S_grid_frame.shape)
    print("experts.S_future_grid_frame.shape", experts.S_future_grid_frame.shape)
    plasma_colors = cm.plasma(np.linspace(0, 1, inference.metadata.T))
    rainbow_colors = cm.plasma(np.linspace(0, 1, inference.metadata.K))
    all_esp_coords = sampled_output.rollout.S_grid_frame
    all_expert_coords = experts.S_future_grid_frame
    esp_coords = None
    expert_coords = None
    # fig, ax = plt.subplots(inference.metadata.B * inference.metadata.A, 3, figsize=(6, 10))
    for b, a in itertools.product(range(inference.metadata.B), range(inference.metadata.A)):

        # ax.tick_params(labelbottom=False, labelleft=False)
        # gridspec inside gridspec
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(4, 3, figure=fig, wspace=0.3, hspace=0.3)
        gs_x_histo = gs[0:2, 0].subgridspec(inference.metadata.T, 1)
        gs_y_histo = gs[2, 1:3].subgridspec(1, inference.metadata.T)
        ax_map     = fig.add_subplot(gs[0:2,1:3])
        ax_map.text(0.05, 0.95, 'trajectories', 
            transform=ax_map.transAxes, ha="left")
        ax_scatter = fig.add_subplot(gs[2,0])
        ax_scatter.text(0.05, 0.9, 'residuals', 
            transform=ax_scatter.transAxes, ha="left")
        ax_boxplot = fig.add_subplot(gs[3,:])
        ax_boxplot.text(0.05, 0.9, 'L^2 error', 
            transform=ax_boxplot.transAxes, ha="left")

        # (K, T, D)
        esp_coords = all_esp_coords[b, :, a, :, :]
        # (T, D)
        expert_coords = all_expert_coords[b, a, :, :]
        # residuals between sampled coords and expert coords
        coords_diff = (esp_coords - expert_coords).reshape(-1, 2)
        print("coords_diff.shape", coords_diff.shape)
        xlim = [np.min(coords_diff[:, 0]), np.max(coords_diff[:, 0])]
        ylim = [np.min(coords_diff[:, 1]), np.max(coords_diff[:, 1])]
        # ax[b + (a*inference.metadata.B), 0].plot(expert_coords[:, 0], expert_coords[:, 1],
        #         markersize=3, linewidth=1, marker='s')
        ax_map.plot(expert_coords[:, 0], expert_coords[:, 1],
                markersize=3, linewidth=1, marker='s')
        for k, color in zip(range(inference.metadata.K), rainbow_colors):
            esp_coords = all_esp_coords[b, k, a, :, :]
            # ax[b + (a*inference.metadata.B), 0].plot(esp_coords[:, 0], esp_coords[:, 1],
            #         markersize=3, color=color, linewidth=1, marker='o', markerfacecolor='none')
            ax_map.plot(esp_coords[:, 0], esp_coords[:, 1],
                    markersize=3, color=color, linewidth=1, marker='o', markerfacecolor='none')

        for t, color in zip(range(inference.metadata.T), plasma_colors):
            esp_coords = all_esp_coords[b, :, a, t, :]
            expert_coords = all_expert_coords[b, a, t, :]
            coords_diff = esp_coords - expert_coords

            # plot error as two dimensional histogram + histogram for per x, y axis
            # ax[b + (a*inference.metadata.B), 1].scatter(coords_diff[:, 0], coords_diff[:, 1],
            #         color=color)
            ax_scatter.scatter(coords_diff[:, 0], coords_diff[:, 1],
                    color=color)
            ax_x_histo = fig.add_subplot(gs_x_histo[inference.metadata.T - t - 1, 0])
            ax_x_histo.hist(coords_diff[:, 0], density=True, color=color)
            ax_x_histo.set_xlim(xlim)
            ax_x_histo.set_ylim([0, 1])
            ax_y_histo = fig.add_subplot(gs_y_histo[0, t])
            ax_y_histo.hist(coords_diff[:, 1], density=True, color=color,
                    orientation="horizontal")
            ax_y_histo.set_ylim(ylim)
            ax_y_histo.set_xlim([0, 1])
            if t != 0:
                ax_x_histo.set_xticks([])
                ax_x_histo.set_yticks([])
                ax_y_histo.set_xticks([])
                ax_y_histo.set_yticks([])
            
            # whisker plots for Euclidean distance
            distances = np.linalg.norm(esp_coords - expert_coords, axis=1)
            # ax[b + (a*inference.metadata.B), 2].scatter(
            #         np.full(distances.shape[0], t), distances,
            #         color=color)
            # ax[b + (a*inference.metadata.B), 2].boxplot(
            #     distances, positions=[t])
            ax_boxplot.scatter(np.full(distances.shape[0], t + 1), distances,
                    s=5, facecolors='none', edgecolors=color)
            ax_boxplot.boxplot(distances, positions=[t + 1 + 0.2])
            ax_boxplot.set_xlim([1, 21])
            ax_boxplot.set_xticks(range(0, inference.metadata.T + 1, 1))
            ax_boxplot.set_xticklabels(
                    [f"T={i}" for i in inference.metadata.T + 1])

        scalarmappaple = cm.ScalarMappable(cmap=cm.plasma)
        scalarmappaple.set_array(inference.metadata.T + 1)
        plt.colorbar(scalarmappaple, ax=ax_boxplot)
        ax_scatter.set_xlim(xlim)
        ax_scatter.set_ylim(ylim)
        gs.tight_layout(fig, pad=1)
        plt.show()
        return

@hydra.main(config_path='conf/esp_infer_config.yaml')
def main(cfg):
    # assert(cfg.main.plot or cfg.main.compute_metrics)
    # output_directory = os.path.realpath(os.getcwd())
    # images_directory = os.path.join(output_directory, 'images')
    # os.mkdir(images_directory)
    log.info("\n\nConfig:\n===\n{}".format(cfg.pretty()))
    # atexit.register(logu.query_purge_directory, output_directory)

    # Instantiate the session.
    sess = tfutil.create_session(
        allow_growth=cfg.hardware.allow_growth,
        per_process_gpu_memory_fraction=cfg.hardware.per_process_gpu_memory_fraction)
    # sess = tf.compat.v1.Session

    # Load the model and the tensor collections.
    log.info("Loading the model...")
    ckpt, graph, tensor_collections = tfutil.load_annotated_model(cfg.model.directory, sess)
    inference = interface.ESPInference(tensor_collections)

    # Instantiate the dataset.
    cfg.dataset.params.T = inference.metadata.T
    cfg.dataset.params.B = inference.metadata.B
    dataset = hydra.utils.instantiate(cfg.dataset, **cfg.dataset.params)

    log.info("Beginning evaluation. Model: {}".format(ckpt))    
    count = 0
    while True:
        # Get the minibatch
        minibatch = dataset.get_minibatch(split=cfg.split,
                input_singleton=inference.training_input,
                is_training=False)
        if minibatch is None:
            break

        sessrun = functools.partial(sess.run, feed_dict=minibatch)
        collect_sample(sessrun, inference)
        return
    
if __name__ == '__main__': main()
