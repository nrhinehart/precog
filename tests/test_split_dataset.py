import pytest
import hydra
import attrdict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

def run_inspect_dataset(cfg):
    dataset = hydra.utils.instantiate(
            cfg.dataset, **cfg.dataset.params)
    # dataset = precog.dataset.split_dataset.SplitDataset(**cfg.dataset.params)
    
    assert dataset.data_path == '/media/external/data/precog_generate/datasets/20210113'
    assert dataset.split_path == '/media/external/data/precog_generate/splits/20210119/12_val0_test1.json'
    assert dataset.name == 'custom_dataset'
    assert dataset.B == 10
    assert 5065 == dataset.split_collections['train'].capacity
    assert 20 == dataset.split_collections['val'].capacity
    assert 566 == dataset.split_collections['test'].capacity
    raw_minibatch = dataset.fetch_raw_minibatch('train', mb_idx=0)
    assert 10 == len(raw_minibatch)
    minibatch = dataset.process_minibatch(raw_minibatch, True)
    minibatch = attrdict.AttrDict(minibatch)
    assert (10, 5, 10, 2) == minibatch.S_past_world_frame.shape
    assert (10, 5, 20, 2) == minibatch.S_future_world_frame.shape
    assert (10, 100, 100, 4) == minibatch.overhead_features.shape
    assert 0 == dataset.split_collections['train'].position

def test_inspect_dataset():
    with initialize(
            config_path='config',
            job_name="testing"):
        cfg = compose(config_name="test_config")
        run_inspect_dataset(cfg)

def run_get_minibatch(cfg):
    dataset = hydra.utils.instantiate(
            cfg.dataset, **cfg.dataset.params)
    minibatch = dataset.get_minibatch(True)
    assert (10, 5, 10, 2) == minibatch.phi.S_past_world_frame.shape
    assert (10, 5, 20, 2) == minibatch.experts.S_future_world_frame.shape
    assert (10, 100, 100, 4) == minibatch.phi.overhead_features.shape
    assert None == dataset.get_minibatch(False, 'val', 9999)
    assert 0 == dataset.split_collections['val'].position

def test_get_minibatch():
    with initialize(
            config_path='config',
            job_name="testing"):
        cfg = compose(config_name="test_config")
        run_get_minibatch(cfg)

def run_exhaust_dataset(cfg):
    dataset = hydra.utils.instantiate(
            cfg.dataset, **cfg.dataset.params)
    assert 20 == dataset.split_collections['val'].capacity
    count = 0
    while True:
        minibatch = dataset.get_minibatch(False, 'val')
        if minibatch is None:
            break
        else:
            count += 1
        if count > 22:
            break
    assert 20 == count
    assert 20 == dataset.split_collections['val'].position
    assert dataset.get_minibatch(False, 'val') is None
    dataset.reset_split('val')
    assert 0 == dataset.split_collections['val'].position
    assert dataset.get_minibatch(False, 'val') is not None
    assert 1 == dataset.split_collections['val'].position

def test_exhaust_minibatch():
    """Test exhausting dataset and reseting dataset works as expected.
    """
    with initialize(
            config_path='config',
            job_name="testing"):
        cfg = compose(config_name="test_config")
        run_exhaust_dataset(cfg)
