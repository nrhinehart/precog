
from abc import ABCMeta, abstractproperty, abstractmethod
import logging
import numpy as np
import os
import six

import precog.utils.class_util as classu

log = logging.getLogger(os.path.basename(__file__))

class SplitMinibatchTracker:
    @classu.member_initialize
    def __init__(self, split_names, split_mb_counts, split_shuffles):
        self.mb_indices = {}
        self.mb_maxes = {}
        for split_name, count in zip(split_names, split_mb_counts):
            self.mb_indices[split_name] = 0
            self.mb_maxes[split_name] = count

    def reset(self, split):
        self.mb_indices[split] = 0

@six.add_metaclass(ABCMeta)    
class MinibatchedDataset:
    @abstractproperty
    def train_data(self):
        pass

    @abstractproperty
    def val_data(self):
        pass

    @abstractproperty
    def test_data(self):
        pass

    def get_minibatch_with_replacement(self, split='train', **kwargs):
        mb_idx = np.random.choice(self.smt.mb_maxes[split])
        return self.get_minibatch(split=split, mb_idx=mb_idx, **kwargs)

    def _index_minibatches(self):
        self.data = {'train': self.train_data, 'val': self.val_data, 'test': self.test_data}        

        self._n_train = len(self.train_data)
        self._n_validation = len(self.val_data)
        self._n_test = len(self.test_data)

        self.train_inds = np.arange(self._n_train)
        self.val_inds = np.arange(self._n_validation)
        self.test_inds = np.arange(self._n_test)

        self.train_mb_count = int(self._n_train / self.B)
        self.val_mb_count = int(self._n_validation / self.B)
        self.test_mb_count = int(self._n_test / self.B)

        self.smt = SplitMinibatchTracker(
            # Names of the splits.
            split_names=['train', 'val', 'test'],
            # Counts of the number of minibatches of a split
            split_mb_counts=[self.train_mb_count, self.val_mb_count, self.test_mb_count],
            # Indicate whether to shuffle each split.
            split_shuffles={'train': True, 'val': False, 'test': False})

    def reset_split(self, split):
        self.smt.reset(split)
        # Shuffle once we're through.
        if self.smt.split_shuffles[split] and hasattr(self, '_rng'): self._rng.shuffle(self.data[split])
