"""
1. First set up dataset directory with sample pattern:
root/map/episode/sample_id
Currently we have 6 maps and 10 episodes.

2. Create 10 groups.
Trivially we can do:
grp1  grp2 ... grp10
m1e1  m1e2 ... m1e10
m2e2  m2e3 ... m2e1
...
m6e10 m6e1 ... m6e9

Create groups by shuffling each map episodes to a group.
This will be a json file where each key represents a group
    and value is a list of sample IDs.

3. Create cross validation splits by combining lists in a group
    together into into (train, val, test) sets.

4. Get dataset loading to work with splits
"""

import os
import glob
import json
import logging
import numpy as np
import utility as util

log = logging.getLogger(os.path.basename(__file__))

def gen_splits(n):
            """Generator of group indices for (train, val, test) set.
            Only yields n-1 of the possible index combinations"""
            v = list(range(10))
            for idx in range(0, len(v) - 1):
                yield tuple(v[:idx] + v[idx + 2:]), (idx,), (idx + 1,)

class ReadPathMixin(object):
    """Mixin to read paths to a list.

    Attributes
    ----------
    data_path : str
        Path of dataset root

    sample_pattern : list of str
        The patch
    
    suffix : str
        File suffix to filter paths by.
    """

    def get_sample_paths(self):
        """Get sample paths from dataset root.

        Returns
        -------
        list of str
            List of patch paths
        """
        patch_path_wildcard = self.data_path
        patterns = sorted([[v, k] for k, v in self.sample_pattern.items()],
                key=lambda x: x[0])
        patterns = map(lambda x: x[1], patterns)
        for word in patterns:
            patch_path_wildcard = os.path.join(patch_path_wildcard, '**')
        patch_path_wildcard = os.path.join(patch_path_wildcard,
                f"*.{self.suffix}")
        patch_paths = glob.glob(os.path.join(patch_path_wildcard))
        return patch_paths


class CrossValidationSplitCreator(object):

    def __init__(self, config):
        self.config = config
        self.save_split_path = config.save_split_path

    def save_split(self, split, n_groups, val, test):
        """
        Parameters
        ----------
        """
        name = f"{n_groups}_val{util.underscore_list(val)}_test{util.underscore_list(test)}"
        fp = os.path.join(os.path.abspath(self.save_split_path), f"{name}.json")
        util.save_json(fp, split)

    def make_splits(self, groups):
        """Make cross validation splits,
        generating groups to make splits if necessary.

        1.
        2.
        
        Splits are JSON dictionaries where
        train set is index 0
        val   set is index 1
        test  set is index 2

        Parameters
        ----------
        groups : dict of key: (list of str)
            Groups of sample IDs.
        """

        n_groups = len(groups)
        for train, val, test in gen_splits(n_groups):
            split = {
                    0: util.merge_list_of_list([groups[idx] for idx in train]),
                    1: util.merge_list_of_list([groups[idx] for idx in val]),
                    2: util.merge_list_of_list([groups[idx] for idx in test])}
            np.random.shuffle(split[0])
            np.random.shuffle(split[1])
            np.random.shuffle(split[2])
            self.save_split(split, n_groups, val, test)


class SampleRetriever(ReadPathMixin):
    
    def __init__(self, config):
        # ReadPathMixin parameters
        self.data_path = os.path.abspath(config.data_path)
        self.sample_pattern = util.create_sample_pattern(
                config.sample_pattern)
        self.suffix = config.suffix
        self.filter_paths = config.filter_paths
        self.sample_paths = self.get_sample_paths()
        
        self.group_by_words = ['map', 'episode', 'agent']
        self.sample_ids = util.create_sample_ids(
                self.sample_paths, sample_pattern=self.sample_pattern)
        self.mapped_ids, self.word_to_labels = util.group_ids(
                self.sample_ids, self.group_by_words, self.sample_pattern)
        for word in self.word_to_labels.keys():
            if word in self.filter_paths:
                self.word_to_labels[word] = util.filter_to_list(
                        lambda label: label in self.filter_paths[word],
                        self.word_to_labels[word])

    def retrieve_loop(self, shuffle=False):
        if shuffle:
            np.random.shuffle(self.word_to_labels['episode'])
            np.random.shuffle(self.word_to_labels['map'])
            np.random.shuffle(self.word_to_labels['agent'])
        for episode in self.word_to_labels['episode']:
            for map_name in self.word_to_labels['map']:
                for agent_name in self.word_to_labels['agent']:
                    try:
                        ids = self.mapped_ids[map_name][episode][agent_name]
                    except KeyError:
                        continue
                    yield (map_name, episode, agent_name), ids


class SampleGroupCreator(SampleRetriever):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if config.n_groups < 3:
            raise ValueError(f"n_groups={config.n_groups} is too small!")
        self.n_groups = config.n_groups
        self.filter_labels = config.filter_labels

    def __should_filter_id(self, iid):
        """
        Note: attribute filter_labels should be non-empty.
        Note: has side effect of filtering corrupted files.
        """
        try:
            filepath = os.path.join(self.data_path, f"{iid}.{self.suffix}")
            datum = util.load_json(filepath)
        except json.decoder.JSONDecodeError as e:
            log.warning(f"corrupted file {filepath} " + repr(e))
            return False
        labels = datum["labels"]
        for label_name, values in self.filter_labels.items():
            if label_name in labels:
                if str(labels[label_name]) not in values:
                    return False
        return True

    def __filter_ids(self, ids):
        if self.filter_labels:
            return util.filter_to_list(self.__should_filter_id, ids)

    def generate_groups(self):
        """Generate groups

        1. an episode in a map is not represented in more than one group.
        2. maps are represented evenly across all groups.
        3. groups are roughly the same size and is shuffled

        Returns
        -------
        dict of int: (list of str)
            Sample IDs grouped into n_groups groups.
        """
        raw_groups = { }
        for idx in range(self.n_groups):
            raw_groups[idx] = [ ]

        # add IDs to list
        log.info("filtering out paths by " + repr(self.filter_paths))
        group_idx = 0
        for labels, ids in self.retrieve_loop(shuffle=True):
            raw_groups[group_idx].append(ids)
            group_idx = (group_idx + 1) % self.n_groups

        # concatenate list of list
        groups = { }
        for idx, raw_group in raw_groups.items():
            groups[idx] = util.merge_list_of_list(raw_group)
            np.random.shuffle(groups[idx])

        for idx, group in groups.items():
            logging.info(f"group {idx} has {len(group)} samples")

        # filter each group if necessary
        if self.filter_labels:
            log.info("filtering out samples by " + repr(self.filter_labels))
            for idx in groups.keys():
                next_group = self.__filter_ids(groups[idx])
                if next_group:
                    groups[idx] = next_group
                else:
                    log.warning("Filtering left out all samples in a group!")
                    raise Exception("Filtering left out all samples in a group!")
        
            for idx, group in groups.items():
                logging.info(f"group {idx} after filtering has {len(group)} samples")

        return groups
    
    def generate_cross_validation_splits(self, groups=None):
        """Make cross validation splits,
        generating groups to make splits if necessary.

        Parameters
        ----------
        groups : dict of key: (list of str)
            Groups of sample IDs.
        """
        if groups is None:
            groups = self.generate_groups()
        cvsc = CrossValidationSplitCreator(self.config)
        cvsc.make_splits(groups)
