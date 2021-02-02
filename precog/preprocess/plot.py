import os
import attrdict
import json
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utility as util
import precog.utils.similarity_util as similarityu
from precog.preprocess import SampleRetriever

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

class PlottingException(Exception):
    pass

def astype(x, dtype):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif hasattr(x, 'numpy'):
        return astype(x.numpy(), dtype)
    else:
        raise ValueError("Unrecognized type")

def load_json(json_fn):
    """Load a json datum.

    :param json_fn: <str> the path to the json datum.
    :returns: dict of postprocess json data.
    """
    assert(os.path.isfile(json_fn))
    with open(json_fn, 'r') as f:
        json_datum = json.load(f)
    postprocessed_datum = from_json_dict(json_datum)
    return postprocessed_datum
    
def from_json_dict(json_datum):
    """Postprocess the json datum to ndarray-ify things

    :param json_datum: dict of the loaded json datum.
    :returns: dict of the postprocessed json datum.
    """
    pp = attrdict.AttrDict()
    for k, v in json_datum.items():
        if isinstance(v, list) or isinstance(v, float):
            pp[k] = np.asarray(v)
        elif isinstance(v, dict) or isinstance(v, int) or isinstance(v, str):
            pp[k] = v
        else:
            raise ValueError("Unrecognized type")
    return pp

save_partial_path = "plots"

def plot_sample(datum, idx, name, figsize=(8, 8)):
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    for ii, ij in [(0, 0), (0, 1), (1, 0), (1, 1)]: 
        pass
    ############
    # reset plot
    ax.cla()
    ax.set_aspect('equal')
    ##########
    # plot bev
    below_slice = slice(2,3) # get index 2
    above_slice = slice(1,2) # get index 1

    bev0 = astype(datum.overhead_features[..., below_slice], np.float64).sum(axis=-1)
    grey_pixels = np.where(bev0 > 0.01)
    bev1 = astype(datum.overhead_features[..., above_slice], np.float64).sum(axis=-1)
    red_pixels = np.where(bev1 > 0.01)
    image = 255 * np.ones(bev0.shape[:2] + (3,), dtype=np.uint8)
    # Grey for below
    image[grey_pixels] = [153, 153, 153]
    # Red for above
    image[red_pixels] = [228, 27, 28]
    ax.imshow(image, origin='upper')
    ############
    # get shapes
    H, W, C = datum.overhead_features.shape
    A, T, _ = datum.agent_futures.shape
    T_past, _ = datum.player_past.shape

    feature_pixels_per_meter=2
    origin = datum.player_past[-1,:2]
    local2world = similarityu.SimilarityTransform.from_origin_and_rotation(
            origin, 0., degrees=True, scale=1., lib=np)
    # world2local = local2world.invert()
    world_origin_grid_frame = np.array([H//2, W//2], dtype=np.float64)
    world2grid = similarityu.SimilarityTransform.from_origin_and_rotation(
            world_origin_grid_frame, 0., degrees=True, scale=feature_pixels_per_meter, lib=np)
    # local2grid = world2grid * local2world

    ###################################
    # plot past and future trajectories
    futures = np.append(datum.player_future[None], datum.agent_futures, axis=0)
    futures = futures[...,:2]
    futures = world2grid.apply(futures)
    for k, future in enumerate(futures):
        future = future.T
        ax.plot(*future, marker="s", linewidth=1, markersize=8, markerfacecolor='none',
                color=COLORS[k], alpha=0.5)
    pasts = np.append(datum.player_past[None], datum.agent_pasts, axis=0)
    pasts = pasts[...,:2]
    pasts = world2grid.apply(pasts)
    for k, past in enumerate(pasts):
        past = past.T
        ax.plot(*past, marker="d", linewidth=1, markersize=8, markerfacecolor='none',
                color=COLORS[k])
    # plt.savefig("{}/overhead_{:05d}.png".format(save_partial_path, idx))
    plt.savefig("{}/{}.png".format(save_partial_path, name))
    # plt.show()
    plt.close('all')

data_dir = "/media/external/data/precog_generate/datasets/20210127"
wildcard = "*.json"
data_file_wildcard = os.path.join(data_dir, wildcard)
data_file_paths = glob.glob(data_file_wildcard)
for idx, p in enumerate(data_file_paths):
    name = os.path.basename(os.path.splitext(p)[0])
    os.path.basename
    datum = load_json(p)
    plot_sample(datum, idx + 1, name)
    # break

def plot_impl(ax, datum):
    ax.set_aspect('equal')
    # road LIDAR points
    bev0 = astype(datum.overhead_features[..., 2], np.float64)
    grey_pixels = np.where(bev0 > 0.01)
    # obstacle LIDAR points
    bev1 = astype(datum.overhead_features[..., 1], np.float64)
    red_pixels = np.where(bev1 > 0.01)
    image = 255 * np.ones(bev0.shape[:2] + (3,), dtype=np.uint8)
    # Grey for below
    image[grey_pixels] = [153, 153, 153]
    # Red for above
    image[red_pixels] = [228, 27, 28]
    ax.imshow(image, origin='upper')
    ############
    # get shapes
    H, W, C = datum.overhead_features.shape
    A, T, _ = datum.agent_futures.shape
    T_past, _ = datum.player_past.shape

    feature_pixels_per_meter=2
    origin = datum.player_past[-1,:2]
    local2world = similarityu.SimilarityTransform.from_origin_and_rotation(
            origin, 0., degrees=True, scale=1., lib=np)
    # world2local = local2world.invert()
    world_origin_grid_frame = np.array([H//2, W//2], dtype=np.float64)
    world2grid = similarityu.SimilarityTransform.from_origin_and_rotation(
            world_origin_grid_frame, 0., degrees=True, scale=feature_pixels_per_meter, lib=np)
    # local2grid = world2grid * local2world

    ###################################
    # plot past and future trajectories
    futures = np.append(datum.player_future[None], datum.agent_futures, axis=0)
    futures = futures[...,:2]
    futures = world2grid.apply(futures)
    for k, future in enumerate(futures):
        future = future.T
        ax.plot(*future, marker="s", linewidth=1, markersize=8, markerfacecolor='none',
                color=COLORS[k], alpha=0.5)
    pasts = np.append(datum.player_past[None], datum.agent_pasts, axis=0)
    pasts = pasts[...,:2]
    pasts = world2grid.apply(pasts)
    for k, past in enumerate(pasts):
        past = past.T
        ax.plot(*past, marker="d", linewidth=1, markersize=8, markerfacecolor='none',
                color=COLORS[k])

def plot_4(datum1, datum2, datum3, datum4, figsize=(16, 16)):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    batch = zip([ax1, ax2, ax3, ax4], [datum1, datum2, datum3, datum4])
    for ax, datum in zip(
            [ax1, ax2, ax3, ax4],
            [datum1, datum2, datum3, datum4]):
        plot_impl(ax, datum)

class InspectSamplePlotter(SampleRetriever):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.save_dir = config.save_plot_path
    
    def __sample_id_to_filepath(self, sample_id):
        return os.path.join(self.data_path,
                f"{sample_id}.{self.suffix}")

    def plot(self):
        for labels, ids in self.retrieve_loop():
            map_name, episode, agent_name = labels
            ii = 0
            try:
                while True:
                    try:
                        ids = random.sample(ids, 4)
                    except ValueError:
                        raise PlottingException()
                    fps = util.map_to_list(self.__sample_id_to_filepath, ids)
                    try:
                        data = util.map_to_list(load_json, fps)
                        break
                    except json.decoder.JSONDecodeError as e:
                        print(f"corrupted file {filename} " + repr(e))
                    ii +=1
                    if ii > 4:
                        raise PlottingException()
            except PlottingException:
                continue

            plot_4(*data)
            savepath = os.path.join(self.save_dir,
                    f"{map_name}_{episode}_{agent_name}.png")
            plt.savefig(savepath)
            plt.close('all')
