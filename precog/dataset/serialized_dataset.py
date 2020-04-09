
import attrdict
import dill
import json
import glob
import logging
import numpy as np
import os
import pdb

import precog.interface as interface
import precog.dataset.metadata_producers as metadata_producers
import precog.dataset.minibatched_dataset as minibatched_dataset
import precog.utils.class_util as classu
import precog.utils.np_util as npu
import precog.utils.log_util as logu

log = logging.getLogger(os.path.basename(__file__))

class SerializedDataset(interface.ESPDataset, minibatched_dataset.MinibatchedDataset):
    input_keys = ['player_future', 'agent_futures', 'player_past', 'agent_pasts', 'player_yaw', 'agent_yaws', "overhead_features"]
    
    @logu.log_wrapi()
    @classu.member_initialize
    def __init__(self,
                 root_path,
                 T,
                 load_bev=True,
                 max_data=int(1e9),
                 B=None,
                 _max_A=5,
                 feature_pixels_per_meter=2.,
                 W=100,
                 fmt='json',
                 T_past=10,
                 match_prefix='ma*',
                 train_suffix='/train/',
                 val_suffix='/val/',
                 test_suffix='/test/',
                 keyremap=None,
                 sdt_bev=False,
                 save_sdts=True,
                 _name='unnamed',
                 extra_params={}):
        train_path = root_path + train_suffix
        val_path = root_path + val_suffix
        test_path = root_path + test_suffix
        assert(fmt in ('json', 'dill'))
        match_str = '/{}{}'.format(match_prefix, fmt)
        self._rng = np.random.default_rng()

        log.debug("Indexing split filenames")
        self._train_data = np.asarray(sorted(glob.glob(train_path + match_str)))[:max_data]
        self._val_data = np.asarray(sorted(glob.glob(val_path + match_str)))[:max_data]
        self._test_data = np.asarray(sorted(glob.glob(test_path + match_str)))[:max_data]
            
        self.Other_count = self.max_A - 1
        self.Other_count_past = self.Other_count
        log.debug("Indexing split statistics")
        self._index_minibatches()

        # Possible shuffle the splits.
        log.debug("Shuffling splits")
        for split in ('train', 'val', 'test'):
            if self.smt.split_shuffles[split]: self._rng.shuffle(self.data[split])

        # Create identity map
        if self.keyremap is None:
            self.keyremap = {}
            for k in self.input_keys:
                self.keyremap[k] = k
        
        assert(len(self.train_data) > 0)
        assert(len(self.val_data) > 0)
        assert(len(self.test_data) > 0)

    @property
    def max_A(self):
        return self._max_A

    @property
    def name(self): return self._name
    @property
    def train_data(self): return self._train_data
    @property
    def val_data(self): return self._val_data
    @property
    def test_data(self): return self._test_data

    def get_T(self): return self.T

    @logu.log_wrapd()
    def get_minibatch(self, is_training, split='train', mb_idx=None, input_singleton=None, *args, **kwargs):
        if mb_idx is None:
            # Use the minibatch tracker to get the current minibatch index for the split
            mb_idx = self.smt.mb_indices[split]
            # Update the state of the split's minibatch tracker
            self.smt.mb_indices[split] += 1
        else:
            # If the mb_idx is provided, don't do any index updating.
            pass

        # If the minibatch tracker is out of bounds, reset the split, and return None to indicate the epoch is over.
        if mb_idx >= self.smt.mb_maxes[split] or mb_idx < 0:
            self.reset_split(split)
            return None

        k = lambda _, _k: getattr(_, self.keyremap[_k])

        data = self._fetch_minibatch(mb_idx, split)
        
        # (1, B, T, d)
        player_experts = np.stack([np.asarray(k(_, 'player_future')) for _ in data], 0)[None]

        # (1, B, Tp, d)
        player_pasts = np.stack([np.asarray(k(_, 'player_past')) for _ in data], 0)[None]
        
        # (1, B)
        player_yaws = np.stack([k(_, 'player_yaw') for _ in data], 0)[None]

        # (O, B, T, d)
        if self.Other_count > 0:
            other_experts = np.stack([npu.fill_axis_to_size(
                np.asarray(k(_, 'agent_futures'))[:self.Other_count], axis=0, size=self.Other_count) for _ in data], 1)
            # (O, B, Tp, d)
            other_pasts = np.stack([npu.fill_axis_to_size(
                np.asarray(k(_, 'agent_pasts'))[:self.Other_count], axis=0, size=self.Other_count_past) for _ in data], 1)
                
            # (O, B)
            other_yaws = np.stack([npu.fill_axis_to_size(
                k(_, 'agent_yaws')[:self.Other_count], axis=0, size=self.Other_count) for _ in data], 1)

            # (A, B, Tpast, d)
            pasts = np.concatenate([player_pasts, other_pasts], 0)[..., -self.T_past:, :]
            experts = np.concatenate([player_experts, other_experts], 0)
            yaws = np.concatenate([player_yaws, other_yaws], 0)
        else:
            # (A, B, Tpast, d)
            pasts = player_pasts[..., -self.T_past:, :]
            experts = player_experts
            yaws = player_yaws
            
        # (O, B)
        # scene_tokens = np.asarray([_.metadata['scene_token'] for _ in data])
        # agent_annotation_tokens = np.stack([npu.fill_axis_to_size(
        #     _.metadata['agent_annotation_tokens'][:self.Other_count], axis=0, size=self.Other_count) for _ in data], 1)

        metadata_list = metadata_producers.PRODUCERS[self._name + '_' + self.fmt](data)
        
        # TODO
        up = lambda x: x.upper() if x else x
        try:
            if self.fmt == 'json':
                # HACK disallow failure only for dill since CARLA data should all be json.
                light_strings = np.asarray([up(_['light_strings']) for _ in data], dtype=np.unicode_)
            else:
                light_strings = np.asarray([up(_.get('traffic_light_state', None)) for _ in data], dtype=np.unicode_)
            if not all(light_strings):

                assert(not any(light_strings))
                light_strings = np.asarray(['NONE' for _ in data])
        except AttributeError as e:
            if self.fmt == 'json': raise AttributeError(e)
            light_strings = np.asarray(['NONE' for _ in data])
            
        if experts.shape[-2] < self.T:
            if not self.extra_params.allow_horizon_mismatch:
                raise ValueError("Experts are too short for requested time horizon!")
            else:
                log.warning("Expert horizon mismatch")
        elif experts.shape[-2] > self.T:
            experts = experts[..., :self.T, :]
            assert(experts.shape[-2] == self.T)            
        else: pass
        
        # (A, B)
        agent_presence = np.zeros((self.max_A, self.B), dtype=np.float32)
        for bi, d in enumerate(data):
            if self.Other_count > 0:
                if isinstance(k(d, 'agent_futures'), np.ndarray):
                    size = k(d, 'agent_futures').shape[0]
                else:
                    size = len(k(d, 'agent_futures'))
                agent_presence[:size + 1, bi] = 1
            else:
                agent_presence[0, bi] = 1
        if self.max_A < agent_presence.shape[0]:
            log.warning("Max A is less than possible A: {} < {}".format(self.max_A, agent_presence.shape[0]))
        agent_presence[self.max_A:] = 0

        # (B, H, W, C), right-handed overhead view.
        if self.load_bev:
            bevs = np.stack([k(_, 'overhead_features') for _ in data], 0)
            bevs = npu.batch_center_crop(bevs, target_h=self.W, target_w=self.W)

        # TODO SDT-ification for CARLA should be handled at data creation time, not loading time.            
        if self.sdt_bev:
            minibatch_filenames = self._fetch_minibatch_filenames(mb_idx, split)
            bevs_sdt = get_sdt(bevs, minibatch_filenames, **self.extra_params.get_sdt_params)
            bevs = bevs_sdt
            
        # Create phi.
        if input_singleton:
            return input_singleton.to_feed_dict(
                S_past_world_frame=npu.frontswap(pasts[..., :2].astype(np.float64)),
                yaws=npu.frontswap(yaws.astype(np.float64)),
                overhead_features=bevs.astype(np.float64),
                agent_presence=npu.frontswap(agent_presence.astype(np.float64)),
                S_future_world_frame=npu.frontswap(experts.astype(np.float64))[..., :2],
                light_strings=light_strings,
                metadata_list=metadata_list,
                is_training=np.array(is_training))
        else:        
            return interface.ESPTrainingInput.from_data(
                S_past_world_frame=npu.frontswap(pasts[..., :2].astype(np.float64)),
                yaws=npu.frontswap(yaws.astype(np.float64)),
                overhead_features=bevs.astype(np.float64),
                agent_presence=npu.frontswap(agent_presence.astype(np.float64)),
                S_future_world_frame=npu.frontswap(experts.astype(np.float64))[..., :2],
                feature_pixels_per_meter=np.asarray(self.feature_pixels_per_meter, np.float64),
                light_strings=light_strings,
                metadata_list=metadata_list,
                is_training=np.array(is_training))

    @logu.log_wrapd(True)
    def _fetch_item(self, frame_index, split):
        if self.fmt == 'json':
            return load_json(self.data[split][frame_index])
        elif self.fmt == 'dill':
            with open(self.data[split][frame_index], 'rb') as f:
                return dill.load(f)
        else:
            raise ValueError("Unrecognized format1")

    def _fetch_item_filename(self, frame_index, split):
        return self.data[split][frame_index]

    def _fetch_minibatch_filenames(self, mb_idx, split):
        return [self._fetch_item_filename(data_ind, split) for data_ind in self._mb_idx_to_data_inds(mb_idx)]

    def _mb_idx_to_data_inds(self, mb_idx):
        return list(range(mb_idx * self.B, (mb_idx + 1) * self.B))

    def _fetch_minibatch(self, mb_idx, split):
        return [self._fetch_item(data_ind, split) for data_ind in self._mb_idx_to_data_inds(mb_idx)]

    def __repr__(self):
        return self.__class__.__name__ + "(root={})".format(self.root_path)

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

def get_sdt(bevs, filename_keys, sdt_clip_thresh, stamp=True, sdt_zero_h=0, sdt_zero_w=0., sdt_params={}, sdt_params_name=''):
    """

    :param bevs: 
    :param filename_keys: 
    :param sdt_clip_thresh: 
    :param stamp: Whether to zero-out a region in the features corresponding to the car.
    :param sdt_params: params to create the SDT
    :param sdt_params_name: extra key part to specify SDT creation method.
    :returns: 
    :rtype: 

    """
    B, H, W, C = bevs.shape
    for b in range(B):
        # Get the filename of this datum, we'll use it as a key.
        datum_filename = filename_keys[b]
        sdt_filename = datum_filename + sdt_params_name + '_sdt.npz'
            
        if os.path.isfile(sdt_filename):
            try: _load_sdt(bevs, sdt_filename, b)
            except ValueError as e:
                log.error("Caught when trying to load: {}".format(e))
                # Remove it so we can recreate it.
                try: os.remove(sdt_filename)
                except: pass
                _create_sdt(
                    bevs=bevs,
                    sdt_filename=sdt_filename,
                    b=b,
                    H=H,
                    W=W,
                    C=C,
                    sdt_clip_thresh=sdt_clip_thresh,
                    stamp=stamp,
                    sdt_zero_h=sdt_zero_h,
                    sdt_zero_w=sdt_zero_w,
                    sdt_params=sdt_params)
        else:
            _create_sdt(
                bevs=bevs,
                sdt_filename=sdt_filename,
                b=b,
                H=H,
                W=W,
                C=C,
                sdt_clip_thresh=sdt_clip_thresh,
                stamp=stamp,
                sdt_zero_h=sdt_zero_h,
                sdt_zero_w=sdt_zero_w,
                sdt_params=sdt_params)
    # Return the output.
    return bevs

def _load_sdt(bevs, sdt_filename, b):
    # Load the SDT into the bevs.
    sdt_b = np.load(sdt_filename, allow_pickle=True)['arr_0']
    bevs[b] = sdt_b

def _create_sdt(bevs, sdt_filename, b, H, W, C, sdt_clip_thresh, stamp=True, sdt_zero_h=0, sdt_zero_w=0., sdt_params={}, save=True):
    # Create the SDT for this batch item.
    sdt_b = bevs[b]
    if stamp:
        # Stamp zeros out around the ego-car.
        center = (H / 2, W / 2 + 1)
        sdt_b[int(np.floor(center[0] - sdt_zero_h / 2)):int(np.ceil(center[0] + sdt_zero_h / 2)),
              int(np.floor(center[1] - sdt_zero_w / 2)):int(np.ceil(center[1] + sdt_zero_w / 2))] = 0.
    for c in range(C):
        float_input_array = sdt_b[..., c]
        binary_input_array = float_input_array > sdt_clip_thresh
        sdt_b[..., c] = npu.signed_distance_transform(binary_input_array, **sdt_params)
    if save:
        # Don't ever overwrite anything. TODO may cause issues during simultaneous training.
        assert(not os.path.isfile(sdt_filename))
        # Save the SDT
        np.savez_compressed(sdt_filename, sdt_b)
