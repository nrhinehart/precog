
import copy
import dill
import hydra
import logging
import numpy as np
import random
import os
import pdb
import pyquaternion
import tqdm

import precog.utils.class_util as classu

import precog.ext.nuscenes.nuscenes as nuscenes_module
import precog.ext.nuscenes.utils.data_classes as dc
import precog.ext.nuscenes.utils.splits
from precog.ext.nuscenes.utils.geometry_utils import transform_matrix
import precog.ext.nuscenes.utils.geometry_utils as gu

np.set_printoptions(suppress=True, precision=4)

log = logging.getLogger(__file__)

class NuscenesConfig:
    # samples are at 2Hz
    sample_period = 2
    # LIDAR is at 20Hz
    lidar_Hz = 20
    sample_frequency = 1./sample_period

    # Predict 4 seconds in the future with 1 second of past.
    past_horizon_seconds = 1
    future_horizon_seconds = 4

    # The number of samples we need.
    future_horizon_length = int(round(future_horizon_seconds * sample_period))
    past_horizon_length = int(round(past_horizon_seconds * sample_period))

    # Minimum OTHER agents visible.
    min_relevant_agents = 1

    # Configuration for temporal interpolation.
    target_sample_period_past = 5
    target_sample_period_future = 5
    target_sample_frequency_past = 1./target_sample_period_past
    target_sample_frequency_future = 1./target_sample_period_future
    target_past_horizon = past_horizon_seconds * target_sample_period_past
    target_future_horizon = future_horizon_seconds * target_sample_period_future
    target_past_times = -1 * np.arange(0, past_horizon_seconds, target_sample_frequency_past)[::-1]
    # Hacking negative zero to slightly positive for temporal interpolation purposes?
    target_past_times[np.isclose(target_past_times, 0.0, atol=1e-5)] = 1e-8

    # The final relative future times to interpolate
    target_future_times = np.arange(target_sample_frequency_future,
                                    future_horizon_seconds + target_sample_frequency_future,
                                    target_sample_frequency_future)
    # How many sweeps of LIDAR to aggregate.
    n_lidar_sweeps = 10
    # Half-width in meters.
    lidar_meters_max = 50
    # Resolution.
    lidar_pixels_per_meter = 2
    # Histogram size threshold.
    lidar_hist_max_per_pixel = 50
    # Vertical histogram bins (they define the right edges of the histogram)
    lidar_zbins = np.array([-3.,   0.0, 1., 2.,  3., 10.])
    # Whether to normalize the histogram bins by the max count per pixel.
    hist_normalize = True
    # Whether to include the sematic map.
    include_semantic_prior = False
        
class NuscenesMultiagentDatum:
    scene = None
    
    @classu.member_initialize
    def __init__(self,
                 player_past,
                 agent_pasts,
                 player_future,
                 agent_futures,
                 player_yaw,
                 agent_yaws,
                 overhead_features,
                 metadata={}):
        pass
    
    @classmethod
    def from_nuscenes_sample(cls, nusc, sample, attribute_names=['vehicle.moving', 'vehicle.stopped'], cfg=None, offset=0.0):
        """

        :param nusc: 
        :param sample: 
        :param attribute_names: 
        :param 'vehicle.stopped']: 
        :param cfg: 
        :param offset: offset in seconds.
        :returns: 
        :rtype: 

        """
        log.debug("Building datum from {}".format(sample['token']))
        assert(cfg is not None)
        sensor_tokens = sample['data']

        lid_channel = 'LIDAR_TOP'
        lidar_token = sensor_tokens[lid_channel]
        lidar_sample_data = nusc.get('sample_data', lidar_token)

        # Unkeyframed lidar.
        lidar_all_future = nuscenes_module.traverse_linked_list(nusc, lidar_sample_data, 'sample_data', 'next')
        lidar_all_past = nuscenes_module.traverse_linked_list(nusc, lidar_sample_data, 'sample_data', 'prev')
        # The list of all lidar sample_data for this scene.
        lidar_all = lidar_all_past + [lidar_sample_data] + lidar_all_future
        # The list of all lidar timestamps for this scene.
        lidar_timestamps = [_['timestamp'] for _ in lidar_all]
        # Lidar timestamps relative to now.
        lidar_timestamps_sample_relative = [(_ - lidar_sample_data['timestamp']) / 1e6 for _ in lidar_timestamps]
        # We'll reset this later, prevent accidentally using it before then.
        del lidar_sample_data

        # Find the nearest lidar index to the provided offset. 
        offset_index = np.argmin(np.abs(np.asarray(lidar_timestamps_sample_relative) - offset))
        # Get the lidar sample_data at this offset.
        lidar_sample_data = lidar_all[offset_index]
        # Get the lidar sample_data token.
        lidar_now_token = lidar_sample_data['token']
        # The timestamp of the offset lidar.
        timestamp_now = lidar_sample_data['timestamp']
        # Lidar timestamps relative to now.
        lidar_timestamps_relative = [(_ - timestamp_now) / 1e6 for _ in lidar_timestamps]

        # Get the extrinsics of the lidar.
        lidar_extrinsics = nusc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
        # Get the translation of the lidar.
        lidar_xyz_translation = np.asarray(lidar_extrinsics['translation'])
        # Get the orientation of the lidar.
        lidar_q = lidar_extrinsics['rotation']

        # Whether to put everything in ego_pose coords or in LIDAR coords.
        ego_pose_coords = True

        # Car frame in downstream application is different than nuscenes car frame.
        permuter = np.array([[0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        if ego_pose_coords:
            # Undo lidar rotation.
            car_from_ref = transform_matrix(lidar_xyz_translation, pyquaternion.Quaternion(lidar_q), inverse=False)
            car_from_ref = permuter @ car_from_ref
        else:
            ref_permute = pyquaternion.Quaternion(matrix=permuter.astype(np.float32))
            pose_permute = {'translation': [0., 0., 0.], 'rotation': ref_permute.elements}

        def get_ego_pose(pose_token):
            return nusc.get('ego_pose', pose_token)

        # World extrinsics + intrinsics of rear axle on driving car.
        ego_pose_now = get_ego_pose(lidar_sample_data['ego_pose_token'])
        
        def transform_box(box, ego_pose):
            """Transform a box to a coordinate system on the ego-car

            :param box: Box in global coordinates.
            :returns: Box in some other frame's coordinates (either current ego_pose or current LIDAR)
            :rtype: 

            """
            
            # If doing ego-coords, just transform 
            if ego_pose_coords:
                box.transform_to_pose(ego_pose)
            else:
                box.transform_to_pose(ego_pose).transform_to_pose(lidar_extrinsics).transform_to_pose(pose_permute)
        
        # Build a box for the ego-car at now. Note this is not really a good 'box' if it's centered at the ego_pose (rear axle)
        ego_box_now = dc.EgoBox.from_pose(ego_pose_now)

        # Unkeyframed ego poses and boxes in global coordinates.
        ego_poses_all = [get_ego_pose(_['ego_pose_token']) for _ in lidar_all]
        ego_boxes_all = [dc.EgoBox.from_pose(_) for _ in ego_poses_all]

        # Offset the interpolation times.
        target_past_times = cfg.target_past_times
        target_future_times = cfg.target_future_times

        if lidar_timestamps_relative[0] > target_past_times[0]:
            log.debug("Not enough past ego poses. Skipping.")
            return -1
        if lidar_timestamps_relative[-1] < target_future_times[-1]:
            log.debug("Not enough future ego poses. Skipping.")
            return 1
        
        # Interpolate boxes at the target times (global coordinates).
        ego_boxes_past_interp = nuscenes_module.interpolate_boxes_to_times(ego_boxes_all, lidar_timestamps_relative, target_past_times)
        ego_boxes_future_interp = nuscenes_module.interpolate_boxes_to_times(ego_boxes_all, lidar_timestamps_relative, target_future_times)
        ego_boxes_interp = ego_boxes_past_interp + ego_boxes_future_interp

        # The "now" box is the most recent past.
        ego_box_now = ego_boxes_past_interp[-1]
        ego_box_yaw = ego_box_now.get_yaw()

        # Ego boxes in ego_pose now or lidar_pose now coordinates.
        _ = [transform_box(_, ego_pose_now) for _ in ego_boxes_interp]

        # Get annotations in current scene that match the relevant attributes.
        annotations = nusc.explorer.get_annotations_with_attribute_names(sample, attribute_names=attribute_names)

        # If not sufficiently populated with others, refuse to create a multiagent datum.
        if len(annotations) < cfg.min_relevant_agents:
            log.debug("Not enough agents in scene. Skipping.")
            return None

        all_agent_pasts = []
        all_agent_futures = []
        all_agent_yaws = []
        agent_dists = []
        agent_annotation_tokens = []
        agent_annotations = []
        
        for ann in annotations:
            # Keyframed annotations.
            anns_future = nuscenes_module.traverse_linked_list(nusc, ann, 'sample_annotation', 'next')
            anns_past = nuscenes_module.traverse_linked_list(nusc, ann, 'sample_annotation', 'prev')
            anns = anns_past + [ann] + anns_future
            ann_timestamps = [nusc.get('sample', _['sample_token'])['timestamp'] for _ in anns]
            ann_timestamps_relative = [(_ - timestamp_now) / 1e6 for _ in ann_timestamps]
            if ann_timestamps_relative[0] > target_past_times[0]:
                log.debug("Not enough agent past annotations. Skipping.")
                continue
            if ann_timestamps_relative[-1] < target_future_times[-1]:
                log.debug("Not enough agent future annotations. Skipping.")
                continue

            # Annotation boxes in global coordinates.
            ann_boxes = [nusc.get_box(_['token']) for _ in anns]
            agent_annotation_tokens.append(ann['token'])
            agent_annotations.append(ann)

            # Interpolate boxes at the target times (global coordinates.)
            ann_boxes_past_interp = nuscenes_module.interpolate_boxes_to_times(ann_boxes, ann_timestamps_relative, target_past_times)
            ann_boxes_future_interp = nuscenes_module.interpolate_boxes_to_times(ann_boxes, ann_timestamps_relative, target_future_times)
            ann_boxes_interp = ann_boxes_past_interp + ann_boxes_future_interp
            ann_box_now = ann_boxes_past_interp[-1]

            # Annotation boxes in ego_pose now or lidar_pose now coordinates.
            _ = [transform_box(_, ego_pose_now) for _ in ann_boxes_interp]
                        
            # Extract the positions.
            all_agent_pasts.append(np.stack([_.center for _ in ann_boxes_past_interp], axis=-2))
            all_agent_futures.append(np.stack([_.center for _ in ann_boxes_future_interp], axis=-2))
            
            # Yaw in degrees.
            all_agent_yaws.append(ann_box_now.get_yaw() * 180 / np.pi)
            # TODO debug ann box distance? is it the same?
            agent_dists.append(np.linalg.norm(ann_box_now.center[:2] - ego_box_now.center[:2]))

        # If not sufficiently populated with others, refuse to create a multiagent datum.
        if len(agent_dists) < cfg.min_relevant_agents: return None

        # Get the point cloud in the LIDAR_TOP coordinate frame.
        try:
            pc, times = dc.LidarPointCloud.from_file_multisweep(nusc,
                                                                sample_rec=None,
                                                                chan=lid_channel,
                                                                ref_chan=lid_channel,
                                                                sample_data_token=lidar_now_token,
                                                                nsweeps=cfg.n_lidar_sweeps)
        except FileNotFoundError as e:
            log.error("Couldn't find LIDAR file! Skipping creating a datum. Error: '{}'".format(e))
            return None

        # Transform it with the actual LIDAR->ego transformation (and composed axis permutation).
        if ego_pose_coords:
            pc.transform(car_from_ref)

        # (Tp, d). The datum's pasts include 'now'
        player_past = np.stack([_.center for _ in ego_boxes_past_interp], axis=-2)
        # (Tf, d)
        player_future = np.stack([_.center for _ in ego_boxes_future_interp], axis=-2)

        # Compute inds to sort agents by their proximity to the ego-vehicle.        
        sorting_inds = np.argsort(agent_dists)

        #todo debug sorting.
        # (A, Tp, d). Agent past trajectories in ego frame at t=now
        agent_pasts = np.stack(all_agent_pasts, axis=0)[sorting_inds]
        # (A, Tf, d). Agent future trajectories in ego frame at t=now
        agent_futures = np.stack(all_agent_futures, axis=0)[sorting_inds]
        # (A,). Agent yaws in ego frame at t=now
        agent_yaws = np.asarray(all_agent_yaws)[sorting_inds]
        # The annotation tokens for the sample.
        agent_annotation_tokens = np.asarray(agent_annotation_tokens)[sorting_inds]

        # TODO ensure point cloud is in ego-frame?
        BEV, xbins, ybins, zbins = xyz_histogram(pc.points.T,
                                                 cfg.lidar_meters_max,
                                                 cfg.lidar_pixels_per_meter,
                                                 cfg.lidar_hist_max_per_pixel,
                                                 zbins=cfg.lidar_zbins,
                                                 hist_normalize=cfg.hist_normalize)
        if cfg.include_semantic_prior:
            # points should be at the center of the histogram cells (offset by half the cellwidth)
            cellwidth = xbins[1] - xbins[0]
            mask = get_mask(nusc, (xbins + cellwidth/2)[:-1], t=ego_pose_now['translation'], ang=ego_box_yaw, scene_token=sample['scene_token'])
            # ??
            mask = np.flipud(mask)
            BEV = np.concatenate((BEV, mask[...,None]), axis=-1)
            
        datum = NuscenesMultiagentDatum(player_past,
                                        agent_pasts,
                                        player_future,
                                        agent_futures,
                                        player_yaw=0.0,
                                        agent_yaws=agent_yaws,
                                        overhead_features=BEV,
                                        metadata={'sample_token': sample['token'],
                                                  'scene_token': sample['scene_token'],
                                                  'lidar_now_token': lidar_now_token,
                                                  'agent_annotation_tokens': agent_annotation_tokens,
                                                  'ego_translation': ego_pose_now['translation'],
                                                  'ego_yaw': ego_box_yaw})
        return datum

def fill_z(forecast_frame_points, mx=1.):
    future = forecast_frame_points
    future_z = np.concatenate((future, mx*np.ones_like(future[:,[0]])), axis=-1).T
    return future_z

def project_to_camera(nusc, forecast_frame_points, scene_token, cam_str, wmax=1600, hmax=900):
    """

    :param nusc: 
    :param forecast_frame_points: (T, d)
    :param scene_token: 
    :param cam_str: 
    :param wmax: 
    :param hmax: 
    :returns: 
    :rtype: 

    """
    
    sample = nusc.get('sample', scene_token)
    sample_data = nusc.get('sample_data', sample['data'][cam_str])
    cal = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    K = np.asarray(cal['camera_intrinsic'])

    if forecast_frame_points.shape[1] == 3:
        future_fore_C_fore_R = forecast_frame_points.T
    else:
        # Fill in z coordinates. 
        future_fore_C_fore_R = fill_z(forecast_frame_points, mx=0.)
        
    # Homog transform from ego to camera.
    cam_from_ego = gu.transform_matrix(cal['translation'], pyquaternion.Quaternion(cal['rotation']), inverse=True)
    
    # Transform points from ego-frame to camera frame.
    future_cam_C_cam_R = ((cam_from_ego[:3,:3] @ future_fore_C_fore_R).T + cam_from_ego[:3, 3]).T

    # Project to the camera.
    points = gu.view_points(future_cam_C_cam_R, view=K, normalize=False)
    nbr_points = points.shape[1]
    z = points[2]
    
    # Normalize the points.
    points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    # Filter.
    # xib = np.logical_and(0 <= points[0], points[0] <= wmax)
    # yib = np.logical_and(0 <= points[1], points[1] <= hmax)
    # In front of camera.
    zib = 0 <= z
    pib = points[:, zib]
    return pib
                
def xyz_histogram(points, 
                  meters_max=50,
                  pixels_per_meter=2,
                  hist_max_per_pixel=25,
                  zbins=np.linspace(-2, 1, 4),
                  hist_normalize=False):
    assert(points.shape[-1] >= 3)
    assert(points.shape[0] > points.shape[1])
    meters_total = meters_max * 2
    pixels_total = meters_total * pixels_per_meter
    xbins = np.linspace(-meters_max, meters_max, pixels_total + 1, endpoint=True)
    ybins = xbins
    # The first left bin edge must match the last right bin edge.
    assert(np.isclose(xbins[0], -1 * xbins[-1]))
    assert(np.isclose(ybins[0], -1 * ybins[-1]))
    
    hist = np.histogramdd(points[..., :3], bins=(xbins, ybins, zbins), normed=False)[0]

    # Clip histogram 
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel

    # Normalize histogram by the maximum number of points in a bin we care about.
    if hist_normalize:
        overhead_splat = hist / hist_max_per_pixel
    else:
        overhead_splat = hist
    return overhead_splat, xbins, ybins, zbins

def get_mask_from_scene_token(nusc, scene_token):
    return nusc.get('map', nusc.get('log', nusc.get('scene', scene_token)['log_token'])['map_token'])['mask']

def get_mask(nusc, x, t, ang, scene_token):
    mask = get_mask_from_scene_token(nusc, scene_token)
    xx, yy = np.meshgrid(x, x)
    points = np.stack((xx, yy),axis=-1).reshape(-1, 2)
    c = np.cos(ang)
    s = np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    points_r = np.dot(R, points.T).T+t[:2]
    bools = mask.is_on_mask(points_r[:,0].ravel(), points_r[:,1].ravel())
    return bools.reshape(xx.shape)

def create_scene_split_indices(nusc, train=0.8, val=0.1):
    scene_inds = np.arange(len(nusc.scene))
    np.random.shuffle(scene_inds)

    n_train = int(round(train * scene_inds.size))
    n_val = int(round(val * scene_inds.size))
    n_test = scene_inds.size - n_train - n_val

    train_inds = scene_inds[:n_train]
    val_inds = scene_inds[n_train:n_train + n_val]
    test_inds = scene_inds[n_train+n_val:]

    assert(train_inds.size == n_train)
    assert(val_inds.size == n_val)
    assert(test_inds.size == n_test)
    return train_inds, val_inds, test_inds

def create_preprocessed_dataset(output_dir,
                                disjoint_scenes=True,
                                official_splits=True,
                                train=0.8,
                                val=0.1,
                                min_A_total=2,
                                val_scene_idx=None,
                                version='v1.0-trainval',
                                dataroot='/home/nrhinehart/data/datasets/nuscenes_full/',
                                offsets=[0.0],
                                n_max=int(1e8),
                                shuffle=False,
                                dry=True):
    assert(not os.path.isdir(output_dir))
    os.mkdir(output_dir)
    log.info("Creating new nuscenes dataset at {}".format(output_dir))
    offsets = sorted(offsets)
    
    nusc = nuscenes_module.NuScenes(version=version, dataroot=dataroot, verbose=True)
    have_val_scene_idx = val_scene_idx is not None
    if official_splits:
        log.info("Using official trainval splits to create train, val, test data")
        assert(disjoint_scenes)
        train_scenes = precog.ext.nuscenes.utils.splits.train
        val_and_test_scenes = precog.ext.nuscenes.utils.splits.val
        Nval = len(val_and_test_scenes)
        val_scenes, test_scenes = val_and_test_scenes[:Nval//2], val_and_test_scenes[Nval//2:]
        train_scene_inds = [nusc.scene_name_to_scene_idx[_] for _ in train_scenes]
        val_scene_inds = [nusc.scene_name_to_scene_idx[_] for _ in val_scenes]
        test_scene_inds = [nusc.scene_name_to_scene_idx[_] for _ in test_scenes]
    elif have_val_scene_idx:
        assert(disjoint_scenes)
        assert(not official_splits)
        if 'mini' in dataroot:
            val_scene_name = precog.ext.nuscenes.utils.splits.mini_val[int(val_scene_idx)]
        else:
            val_scene_name = precog.ext.nuscenes.utils.splits.val[int(val_scene_idx)]
        scene_idx = nusc.scene_name_to_scene_idx[val_scene_name]
        train_scene_inds = []
        val_scene_inds = []
        test_scene_inds = [scene_idx]
        # train_scenes = val_scenes = val_scenes = nusc.get_scene_samples(val_scene_name)
    else:
        log.info("Creating our own trainval splits")
        train_scene_inds, val_scene_inds, test_scene_inds = create_scene_split_indices(nusc, train=train, val=val)

    train_scenes = [nusc.scene_idx_to_scene_name[_] for _ in train_scene_inds]
    val_scenes = [nusc.scene_idx_to_scene_name[_] for _ in val_scene_inds]
    test_scenes = [nusc.scene_idx_to_scene_name[_] for _ in test_scene_inds]
    scene_splits = {'train_inds': train_scene_inds,
                    'val_inds': val_scene_inds,
                    'test_inds': test_scene_inds,
                    'train_scenes': train_scenes,
                    'val_scenes': train_scenes,
                    'test_scenes': test_scenes}

    log.info("Have {} train, {} val, and {} test scenes".format(len(train_scene_inds), len(val_scene_inds), len(test_scene_inds)))

    os.mkdir(output_dir + '/train/')
    os.mkdir(output_dir + '/val/')
    os.mkdir(output_dir + '/test/')

    def scene_to_sample_tokens(scene):
        samples = nuscenes_module.traverse_linked_list(nusc, nusc.get('sample', scene['first_sample_token']), 'sample', 'next', inclusive=True)
        return [_['token'] for _ in samples]

    train_sample_tokens = []
    val_sample_tokens = []
    test_sample_tokens = []
    
    for ti in train_scene_inds:
        train_sample_tokens.extend(scene_to_sample_tokens(nusc.scene[ti]))
    for vi in val_scene_inds:
        val_sample_tokens.extend(scene_to_sample_tokens(nusc.scene[vi]))
    for tei in test_scene_inds:
        test_sample_tokens.extend(scene_to_sample_tokens(nusc.scene[tei]))

    if not disjoint_scenes:
        assert(not have_val_scene_idx)
        tr_orig = copy.copy(train_sample_tokens)
        va_orig = copy.copy(val_sample_tokens)
        te_orig = copy.copy(test_sample_tokens)
        log.info("Not using disjoint scenes")
        
        assert(not official_splits)
        all_tokens = train_sample_tokens + val_sample_tokens + test_sample_tokens
        if shuffle:
            log.info("Shuffling sample tokens")
            random.shuffle(all_tokens)
        else:
            log.info("Not shuffling sample tokens")

        n_train = int(round(train * len(all_tokens)))
        n_val = int(round(val * len(all_tokens)))
        n_test = len(all_tokens) - n_train - n_val

        del train_sample_tokens, val_sample_tokens, test_sample_tokens
        train_sample_tokens = all_tokens[:n_train]
        val_sample_tokens = all_tokens[n_train:n_train + n_val]
        test_sample_tokens = all_tokens[n_train+n_val:]
        assert(len(test_sample_tokens) == n_test)
        assert(len(val_sample_tokens) == n_val)
        assert(len(train_sample_tokens) == n_train)
        tr = set(train_sample_tokens)
        va = set(val_sample_tokens)
        te = set(test_sample_tokens)
        assert(len(tr & va) == 0)
        assert(len(tr & te) == 0)
        assert(len(va & te) == 0)
        assert(tr | va | te == set(all_tokens))

    # Instantiate the config.
    cfg = NuscenesConfig()
    # Set the number of agents we'll require.
    cfg.min_relevant_agents = min_A_total - 1
    
    with open(output_dir + '/scene_splits.dill', 'wb') as f: dill.dump(scene_splits, f)
    with open(output_dir + '/nuscenes_config.dill', 'wb') as f: dill.dump(cfg, f)

    log.info("Preprocessing train")
    cnt = 0
    train_scene_tokens, val_scene_tokens, test_scene_tokens = [], [], []
    for tok in tqdm.tqdm(train_sample_tokens[:n_max]):
        for offset in offsets:
            datum = NuscenesMultiagentDatum.from_nuscenes_sample(nusc, nusc.get('sample', tok), cfg=cfg, offset=offset)
            if datum in (None, -1): continue
            elif datum == 1: break
            else: pass
                
            train_scene_tokens.append(datum.metadata['scene_token'])
            if not dry:
                with open(output_dir + '/train/ma_datum_{:06d}.dill'.format(cnt), 'wb') as f: dill.dump(datum, f)
            cnt += 1

    log.info("Preprocessing val")        
    cnt = 0
    for tok in tqdm.tqdm(val_sample_tokens[:n_max]):
        for offset in offsets:
            datum = NuscenesMultiagentDatum.from_nuscenes_sample(nusc, nusc.get('sample', tok), cfg=cfg, offset=offset)
            if datum in (None, -1): continue
            elif datum == 1: break
            else: pass
                
            val_scene_tokens.append(datum.metadata['scene_token'])
            if not dry:
                with open(output_dir + '/val/ma_datum_{:06d}.dill'.format(cnt), 'wb') as f: dill.dump(datum, f)
            cnt += 1

    log.info("Preprocessing test")                
    cnt = 0
    for tok in tqdm.tqdm(test_sample_tokens[:n_max]):
        for offset in offsets:
            datum = NuscenesMultiagentDatum.from_nuscenes_sample(nusc, nusc.get('sample', tok), cfg=cfg, offset=offset)
            if datum in (None, -1): continue
            elif datum == 1: break
            else: pass
            test_scene_tokens.append(datum.metadata['scene_token'])
            if not dry:
                with open(output_dir + '/test/ma_datum_{:06d}.dill'.format(cnt), 'wb') as f: dill.dump(datum, f)
            cnt += 1

    tr = set(train_scene_tokens)
    va = set(val_scene_tokens)
    te = set(test_scene_tokens)
    if disjoint_scenes:
        # Ensure scenes disjoint.
        assert(len(tr & va) == 0)
        assert(len(tr & te) == 0)
        assert(len(va & te) == 0)
    else:
        # Ensure scene overlap between each set.
        assert(len(tr & va) > 0)
        assert(len(tr & te) > 0)
        assert(len(va & te) > 0)


@hydra.main(config_path='./preprocess_nuscenes_conf.yaml')
def main(cfg):
    assert(len(cfg.output_dir) > 0)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    assert(0.0 <= min(cfg.offsets) <= max(cfg.offsets) < 0.5)
    create_preprocessed_dataset(cfg.output_dir,
                                official_splits=cfg.official_splits,
                                disjoint_scenes=cfg.disjoint_scenes,
                                min_A_total=cfg.A,
                                version=cfg.version,
                                dataroot=cfg.dataroot,
                                offsets=cfg.offsets,
                                val_scene_idx=cfg.val_scene_idx,
                                train=cfg.train,
                                val=cfg.val,
                                dry=cfg.dry,
                                n_max=cfg.n_max)

if __name__ == '__main__':
    main()
