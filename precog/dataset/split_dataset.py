import os
import logging
import numpy as np
import utility as util

import precog.interface as interface
import precog.dataset.metadata_producers as metadata_producers
import precog.utils.class_util as classu
import precog.utils.log_util as logu

log = logging.getLogger(os.path.basename(__file__))

def batch_center_crop(arr, target_h, target_w):
    *_, h, w, _ = arr.shape
    center = (h // 2, w // 2)
    return arr[..., center[0] - target_h // 2:center[0] + target_h // 2, center[1] - target_w // 2:center[1] + target_w // 2, :]

class MinibatchException(Exception):
        pass

class MinibatchIndexException(MinibatchException):
        pass

class MinibatchCollection(object):
    """Does minibatch index tracking and loading of raw sample data.
    Responsible for figuring out which next minibatch to retrieve."""

    @classu.member_initialize
    def __init__(self, data_path, sample_ids, suffix,
            batch_size, cap=None, shuffle=False):
        """
        """
        if cap is None:
            self.__cap = len(self.sample_ids) // self.batch_size
        else:
            self.__cap = min(cap, len(self.sample_ids) // self.batch_size)
        self.__mb_idx = 0
        if self.shuffle:
            self.reshuffle()
    
    @property
    def position(self):
        """Current index to minibatch in collection sequence."""
        return self.__mb_idx

    @property
    def capacity(self):
        return self.__cap

    def __mb_idx_to_data_indices(self, mb_idx):
        return list(range(mb_idx * self.batch_size, (mb_idx + 1) * self.batch_size))

    def __sample_idx_to_filename(self, sample_idx):
        return os.path.join(self.data_path,
                f"{self.sample_ids[sample_idx]}.{self.suffix}")
    
    def __sample_idx_to_data(self, sample_idx):
        return util.load_json(self.__sample_idx_to_filename(sample_idx))

    def reset(self):
        self.__mb_idx = 0
    
    def reshuffle(self):
        np.random.shuffle(self.sample_ids)

    def fetch(self, mb_idx=None):
        """Fetch raw minibatch data.

        Parameters
        ----------
        mb_idx : int, optional
            Minibatch index.
            If provided then get the minibatch at mb_idx position in collection.
            Otherwise get the next conscutive minibatch in collection sequence.
        
        Returns
        -------
        list of dict
            dict of size

        Raises
        ------
        MinibatchIndexException
            When there are no more minibatches in sequence,
            or mb_idx is larger than capacity.
        """
        if mb_idx is None:
            curr_mb_idx = self.__mb_idx
        else:
            curr_mb_idx = mb_idx
        if curr_mb_idx >= self.capacity:
            raise MinibatchIndexException("ran out of minibatches or index out of range.")
        data_indices = self.__mb_idx_to_data_indices(curr_mb_idx)
        raw_minibatch = util.map_to_list(self.__sample_idx_to_data, data_indices)
        if mb_idx is None:
            self.__mb_idx += 1
        return raw_minibatch
        

class SplitDataset(object):
    """Dataset corresponding to splits.

    Workflow

    1. load split file and construct sample paths from the sample IDs.
    2. when get_minibatch() is called then construct path from sample ID
       and load the JSON sample data from the path into ESPPhi instance.

    Assumptions

    1. assumes that the splits are already shuffled and does not attempt to shuffle.
    2. assumes that ndarrays in sample data are the right format that we want

    """

    SPLIT_INDICES = {
            'train': '0',
            'val':   '1',
            'test':  '2'}

    @logu.log_wrapi()
    @classu.member_initialize
    def __init__(self, data_path, split_path, name, B, A, T, W,
            suffix='json', feature_pixels_per_meter=2.,
            train_cap=None, val_cap=None, test_cap=None,
            **kwargs):
        """Initialize

        Parameters
        ----------
        data_path : str
            The dataset root directory
        split_path : str
            Path to split file.
        name : str
            Name of the dataset we are loading.
        B : int
            Batch size
        suffix : str,optional
            Usually json
        """
        self.data_path = os.path.abspath(self.data_path)
        self.split_path = os.path.abspath(self.split_path)
        self.split = util.load_json(self.split_path)
        train_ids = self.split[self.SPLIT_INDICES['train']]
        val_ids   = self.split[self.SPLIT_INDICES['val']]
        test_ids  = self.split[self.SPLIT_INDICES['test']]

        self.split_collections = {
            'train': MinibatchCollection(
                    self.data_path, train_ids, self.suffix, self.B,
                    cap=self.train_cap),
            'val': MinibatchCollection(
                    self.data_path, val_ids, self.suffix, self.B,
                    cap=self.val_cap),
            'test': MinibatchCollection(
                    self.data_path, test_ids, self.suffix, self.B,
                    cap=self.test_cap)}

    def reset_split(self, split):
        self.split_collections[split].reset()

    def process_minibatch(self, raw_minibatch, is_training):
        """Process the minibatch before handing it to ESPPhi

        Based on SerializedDataset.get_minibatch() method.
        Assumes that variables B, A, T, T_past, D, H, W, C
        of samples correspond to the experiment.

        Parameters
        ----------
        raw_minibatch : list of object
            A list of sample data serialized from JSON representing a batch of size B.
        is_training : bool
            Is this minibatch for training?
        """
        
        # player_future data is (T, D)
        # player_future (B, 1, T, D)
        player_future = np.asarray(
                util.map_to_list(lambda data: data['player_future'], raw_minibatch),
                dtype=np.float64)
        player_future = np.swapaxes(player_future[None], 0, 1)
        
        # player_past data is (T_past, D)
        # player_past is (B, 1, T_past, D)
        player_past = np.asarray(
                util.map_to_list(lambda data: data['player_past'], raw_minibatch),
                dtype=np.float64)
        player_past = np.swapaxes(player_past[None], 0, 1)

        # player_yaw data is (1,)
        # player_yaw is (B, 1)
        try:
            player_yaw = np.asarray(
                    util.map_to_list(lambda data: data['player_yaw'], raw_minibatch),
                    dtype=np.float64)
            player_yaw = np.swapaxes(player_yaw[None], 0, 1)
        except:
            player_yaw = np.zeros(player_future.shape[:2], dtype=np.float64)
        
        # SerializedDataset uses precog.utils.np_util.fill_axis_to_size
        #     to fill or clip agent_* matrix
        # agent_futures data is (A-1, T, D)
        # agent_futures is (B, A-1, T, D)
        agent_futures = np.asarray(
                util.map_to_list(lambda data: data['agent_futures'], raw_minibatch),
                dtype=np.float64)
        
        # agent_pasts data is (A-1, T_past, D)
        # agent_pasts is (B, A-1, T_past, D)
        agent_pasts = np.asarray(
                util.map_to_list(lambda data: data['agent_pasts'], raw_minibatch),
                dtype=np.float64)
        
        # agent_yaws data is (A-1,)
        # agent_yaws is (B, A-1)
        try:
            agent_yaws = np.asarray(
                util.map_to_list(lambda data: data['agent_yaws'], raw_minibatch),
                dtype=np.float64)
        except:
            agent_yaws = np.zeros(agent_futures.shape[:2], dtype=np.float64)
        
        # pasts shape is (B, A, T_past, D)
        pasts = np.concatenate((player_past, agent_pasts,), axis=1)
        # experts shape is (B, A, T, D)
        experts = np.concatenate((player_future, agent_futures,), axis=1)
        # yaws shape is (B, A)
        yaws = np.concatenate((player_yaw, agent_yaws,), axis=1)
        # bevs shape is (B, H, W, C)
        bevs = np.asarray(
                util.map_to_list(lambda data: data['overhead_features'], raw_minibatch),
                dtype=np.float64)
        bevs = batch_center_crop(bevs, self.W, self.W)

        # extra unused(?) features
        metadata_list = interface.MetadataList()
        light_strings = np.array(['NONE'] * experts.shape[0], dtype=np.dtype('U'))
        agent_presence = np.ones(experts.shape[:2], dtype=np.float32)
        is_training = np.array(is_training)

        return {
                'S_past_world_frame': pasts,
                'S_future_world_frame': experts,
                'overhead_features': bevs,
                'agent_presence': agent_presence,
                'yaws': yaws,
                'light_strings': light_strings,
                'metadata_list': metadata_list,
                'is_training': is_training}

    def fetch_raw_minibatch(self, split, mb_idx=None):
        return self.split_collections[split].fetch(mb_idx=mb_idx)

    def get_minibatch(self, is_training, split='train',
            mb_idx=None, input_singleton=None):
        """Get a minibatch.

        Parameters
        ----------
        mb_idx : int, optional
            The index of the minibatch in the split to use.
            Don't pass to get the next minibatch in sequence.
        input_singleton: precog.interface.ESPTrainingInput, optional
            ???

        Returns
        -------
        precog.interface.ESPTrainingInput or precog.utils.tfutil.FeedDict
            If input_singleton is provided then return the training input and expert tensors to add to TF's computation graph.
            Otherwise return a feed dict to pass to tf.Session.run()
        """
        # feature_pixels_per_meter=np.asarray(self.feature_pixels_per_meter, np.float64),
        try:
            # JSON object of the sample data for a batch.
            raw_minibatch = self.fetch_raw_minibatch(split, mb_idx=mb_idx)
        except MinibatchIndexException:
            return None
        phi_kwargs = self.process_minibatch(raw_minibatch, is_training)
        if input_singleton is None:
            feature_pixels_per_meter=np.asarray(
                    self.feature_pixels_per_meter, np.float64)
            return interface.ESPTrainingInput.from_data(
                    feature_pixels_per_meter=feature_pixels_per_meter,
                    **phi_kwargs)
        else:
            return input_singleton.to_feed_dict(**phi_args)
