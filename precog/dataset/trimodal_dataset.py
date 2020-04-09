
import collections
import numpy as np
import pdb
import string
import random

import dill
import glob
import logging
import numpy as np
import os
import pdb
import random

import precog.interface as interface
import precog.dataset.metadata_producers as metadata_producers
import precog.dataset.minibatched_dataset as minibatched_dataset
import precog.utils.class_util as classu
import precog.utils.np_util as npu
import precog.utils.log_util as logu

log = logging.getLogger(os.path.basename(__file__))

# For T=20
SPEED = 1
P0 = np.asarray([0, 0], dtype=np.float64)        

TrimodalDatum = collections.namedtuple('TrimodalDatum', ['bev', 'past', 'future'])

class TrimodalDataset(interface.ESPDataset, minibatched_dataset.MinibatchedDataset):
    @logu.log_wrapi()
    @classu.member_initialize
    def __init__(self, T, T_past, C=3, H=100, B=None, perturb_epsilon=1e-2, feature_pixels_per_meter=2, _max_A=1, _name="trimodal_dataset"):
        self.lane_width = round(0.09 * self.H)
        self.lane_width = int(self.lane_width + self.lane_width % 2)

        # Demonstrations only look reasonable for T=20.
        assert(T == 20)
        
        s = self
        
        straight = build_tee_coast(s.H, s.H, s.C, s.lane_width, s.T, s.T_past)    
        left = build_tee_coast_and_left(s.H, s.H, s.C, s.lane_width, s.T, s.T_past, straight_denom=2., turning_radius=T//3 * SPEED * np.pi/4)
        right = build_tee_coast_and_right(s.H, s.H, s.C, s.lane_width, s.T, s.T_past, straight_denom=2., turning_radius=T//3 * SPEED * np.pi/4)

        N_dup_train = self.B // 3 + 1
        N_dup_val = self.B // 3 + 1
        N_dup_test = self.B // 3 + 1

        self._train_data = [straight] * N_dup_train + [left] * N_dup_train + [right] * N_dup_train
        self._val_data = [straight] * N_dup_val + [left] * N_dup_val + [right] * N_dup_train
        self._test_data = [straight] * N_dup_test + [left] * N_dup_test + [right] * N_dup_train
        random.shuffle(self.train_data)
        random.shuffle(self.val_data)
        random.shuffle(self.test_data)

        self._index_minibatches()

    @property
    def max_A(self): return self._max_A
    @property
    def name(self): return 'trimodal'
    @property
    def train_data(self): return self._train_data
    @property
    def val_data(self): return self._val_data
    @property
    def test_data(self): return self._test_data

    def get_T(self): return self.T

    @logu.log_wrapd()
    def get_minibatch(self, mb_idx=None, split='train', input_singleton=None, is_training=False, *args, **kwargs):
        if mb_idx is None:
            # Increment the tracker if a minibatch is not specified.
            mb_idx = self.smt.mb_indices[split]
            self.smt.mb_indices[split] += 1
        else:
            pass

        if mb_idx >= self.smt.mb_maxes[split] or mb_idx < 0:
            self.smt.reset(split)
            return None

        data = self._fetch_minibatch(mb_idx, split)
        bevs = np.stack([_[0] for _ in data], 0)
        pasts = np.stack([_[1] for _ in data], 0)[None]
        experts = np.stack([_[2] for _ in data], 0)[None]
        yaws = np.stack([0. for _ in data], 0)[None]
        agent_presence = np.zeros((1, self.B), dtype=np.float64)
        metadata_list = metadata_producers.PRODUCERS[self._name](data)
        light_strings = np.array(['NONE']*len(data), dtype=np.unicode_)

        if input_singleton:
            return input_singleton.to_feed_dict(
                S_past_world_frame=npu.frontswap(pasts[..., :2].astype(np.float64)),
                yaws=npu.frontswap(yaws.astype(np.float64)),
                overhead_features=bevs.astype(np.float64),
                agent_presence=npu.frontswap(agent_presence.astype(np.float64)),
                light_strings=light_strings,
                S_future_world_frame=npu.frontswap(experts.astype(np.float64))[..., :2],
                metadata_list=metadata_list,
                is_training=np.array(is_training))
        else:
            return interface.ESPTrainingInput.from_data(
                S_past_world_frame=npu.frontswap(pasts[..., :2].astype(np.float64)),
                yaws=npu.frontswap(yaws.astype(np.float64)),
                overhead_features=bevs.astype(np.float64),
                agent_presence=npu.frontswap(agent_presence.astype(np.float64)),
                S_future_world_frame=npu.frontswap(experts.astype(np.float64))[..., :2],
                light_strings=light_strings,
                is_training=np.array(is_training),
                feature_pixels_per_meter=np.asarray(self.feature_pixels_per_meter, np.float64),
                metadata_list=metadata_list)

    @logu.log_wrapd(True)
    def _fetch_item(self, frame_index, split):
        return self.data[split][frame_index]

    def _mb_idx_to_data_inds(self, mb_idx):
        return list(range(mb_idx * self.B, (mb_idx + 1) * self.B))

    def _fetch_minibatch(self, mb_idx, split):
        return [self._fetch_item(data_ind, split) for data_ind in self._mb_idx_to_data_inds(mb_idx)]

    def __repr__(self):
        return self.__class__.__name__ + "()"

# def build_past(p0, pT_past, T_past):
#     x_past = np.linspace(pT_past[0], p0[0], T_past)
#     y_past = np.linspace(pT_past[1], p0[1], T_past)
#     return np.stack((x_past, y_past), axis=-1)

def build_past(p0, speed, T_past):
    dist = T_past * speed
    x_past = np.linspace(-dist, p0[0], T_past)
    y_past = np.linspace(0, p0[1], T_past)
    return np.stack((x_past, y_past), axis=-1)

def rollout_actions(accels, xt, xtm1, T):
    traj = []
    for accel in accels:
        xtp1 = verlet(xt, xtm1, accel)
        traj.append(xtp1)
        xtm1 = xt
        xt = xtp1
    traj = np.asarray(traj)
    assert(traj.shape[0] == T)
    return traj

def verlet(xt, xtm1, xtdotdot):
    return 2*xt - xtm1 + xtdotdot

def standard_coast(T, past):
    return rollout_actions(no_acceleration(T), xt=past[-1], xtm1=past[-2], T=T)

def no_acceleration(T):
    return constant_acceleration(np.zeros((2,), dtype=np.float64), T=T)

def constant_acceleration(a, T):
    assert(a.shape == (2,))
    return np.tile(a[None], (T, 1)).astype(np.float64)

def inv_parametric_circle(x, y, xc, yc):
    t = np.arctan2(y - yc, x - xc)
    return t

def parametric_circle(t, xc, yc, r):
    x = xc + r * np.cos(t)
    y = yc + r * np.sin(t)
    return x, y

def arc(center, start, end, T):
    start = np.asarray(start)
    end = np.asarray(end)
    center = np.asarray(center)
    r = np.linalg.norm(start - center)
    assert(np.isclose(r, np.linalg.norm(end - center)))
    t_start = inv_parametric_circle(start[0], start[1], center[0], center[1])
    t_end = inv_parametric_circle(end[0], end[1], center[0], center[1])
    t_space = np.linspace(t_start, t_end, T, dtype=np.float64)
    return np.stack(parametric_circle(t_space, center[0], center[1], r), axis=-1)

def turn_left(start, T, turning_radius):
    center = start + np.asarray([0, -turning_radius], dtype=np.float64)
    end = center + np.asarray([turning_radius, 0], dtype=np.float64)
    return arc(center=center, start=start, end=end, T=T)

def turn_right(start, T, turning_radius):
    center = start + np.asarray([0, turning_radius], dtype=np.float64)
    end = center + np.asarray([turning_radius, 0], dtype=np.float64)
    return arc(center=center, start=start, end=end, T=T)

def coast_and_turn_left(T, past, turning_radius=12, straight_denom=2):
    T_left = T // 3
    T_coast = int((T - T_left) // straight_denom)
    T_coast2 = T - T_coast - T_left + 1

    coast_actions = no_acceleration(T_coast)
    coast2_actions = no_acceleration(T_coast2)
    
    # left_actions = accelerations_of_left_turn(T=T_left, turning_radius=turning_radius)
    traj = rollout_actions(coast_actions, xt=past[-1], xtm1=past[-2], T=T_coast)
    left_turn = turn_left(traj[-1], T=T_left, turning_radius=turning_radius)

    faux_tm1 = left_turn[-2].copy()
    faux_tm1[0] = left_turn[-1][0]
    traj2 = rollout_actions(coast2_actions, xt=left_turn[-1], xtm1=faux_tm1, T=T_coast2)
    return np.concatenate((traj, left_turn[1:], traj2), axis=0)

def coast_and_turn_right(T, past, turning_radius=12, straight_denom=2):
    T_right = T // 3
    T_coast = int((T - T_right) // straight_denom)
    T_coast2 = T - T_coast - T_right + 1

    coast_actions = no_acceleration(T_coast)
    coast2_actions = no_acceleration(T_coast2)
    
    # right_actions = accelerations_of_right_turn(T=T_right, turning_radius=turning_radius)
    traj = rollout_actions(coast_actions, xt=past[-1], xtm1=past[-2], T=T_coast)
    right_turn = turn_right(traj[-1], T=T_right, turning_radius=turning_radius)

    faux_tm1 = right_turn[-2].copy()
    faux_tm1[0] = right_turn[-1][0]
    traj2 = rollout_actions(coast2_actions, xt=right_turn[-1], xtm1=faux_tm1, T=T_coast2)
    return np.concatenate((traj, right_turn[1:], traj2), axis=0)

def build_tee_coast(H, W, C, lane_width, T, T_past):
    bev = tee(H=H, W=W, C=C, lane_width=lane_width)
    past = build_past(P0, SPEED, T_past)
    future = standard_coast(T=T, past=past)
    return TrimodalDatum(bev, past, future)

def build_tee_coast_and_left(H, W, C, lane_width, T, T_past, **kwargs):
    bev = tee(H=H, W=W, C=C, lane_width=lane_width)
    past = build_past(P0, SPEED, T_past)
    future = coast_and_turn_left(T=T, past=past, **kwargs)
    return TrimodalDatum(bev, past, future)

def build_tee_coast_and_right(H, W, C, lane_width, T, T_past, **kwargs):
    bev = tee(H=H, W=W, C=C, lane_width=lane_width)
    past = build_past(P0, SPEED, T_past)
    future = coast_and_turn_right(T=T, past=past, **kwargs)
    return TrimodalDatum(bev, past, future)

def build_tee_left_coast(H, W, C, lane_width, T, T_past):
    bev = tee_left(H=H, W=W, C=C, lane_width=lane_width)
    past = build_past(P0, SPEED, T_past)
    future = standard_coast(T=T, past=past)
    return TrimodalDatum(bev, past, future)

def build_tee_left_coast_and_left(H, W, C, lane_width, T, T_past, **kwargs):
    bev = tee_left(H=H, W=W, C=C, lane_width=lane_width)
    past = build_past(P0, SPEED, T_past)
    future = coast_and_turn_left(T=T, past=past, **kwargs)
    return TrimodalDatum(bev, past, future)

# BEV creation.
def carve_straightaway(bev, H, W, lane_width, channel=0):
    bev[H // 2 - lane_width - lane_width // 2:H // 2 + lane_width - lane_width // 2] = 1

def carve_left(bev, H, W, lane_width, channel=0):
    bev[0:H // 2, W // 2 + W // 4 - lane_width:W // 2 + W // 4 + lane_width] = 1

def carve_right(bev, H, W, lane_width, channel=0):
    bev[H // 2:, W // 2 + W // 4 - lane_width:W // 2 + W // 4 + lane_width] = 1
    
def tee_left(H, W, C, lane_width):
    bev = np.zeros((H,W), dtype=np.bool)
    carve_straightaway(bev, H=H, W=W, lane_width=lane_width)
    carve_left(bev, H=H, W=W, lane_width=lane_width)
    bev = rasterize_road(bev)
    return bev

def tee_right(H, W, C, lane_width):
    bev = np.zeros((H,W), dtype=np.bool)
    carve_straightaway(bev, H=H, W=W, lane_width=lane_width)
    carve_right(bev, H=H, W=W, lane_width=lane_width)

    bev = rasterize_road(bev)
    return bev  

def tee(H, W, C, lane_width):
    bev = np.zeros((H,W), dtype=np.bool)
    carve_straightaway(bev, H=H, W=W, lane_width=lane_width)
    carve_right(bev, H=H, W=W, lane_width=lane_width)
    carve_left(bev, H=H, W=W, lane_width=lane_width)

    bev = rasterize_road(bev)
    return bev

def rasterize_road(bev):
    raster = np.zeros(shape=bev.shape + (3,), dtype=np.float64)
    road_inds = np.where(bev)
    not_road_inds = np.where(np.logical_not(bev))
    raster[road_inds] = [1., 1., 1.]
    raster[not_road_inds] = [0.2, .4, 0.]
    return raster
