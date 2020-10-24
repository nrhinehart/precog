
import numpy as np
import tensorflow as tf

import precog.interface as interface

def nuscenes_dill_metadata_producer(data):
    items = interface.MetadataList()
    sample_tokens = np.asarray([_.metadata['scene_token'] for _ in data])
    items.append(interface.MetadataItem(name='sample_token', array=sample_tokens, dtype=tf.string))
    # scene_tokens = np.asarray([_.metadata['real_scene_token'] for _ in data])
    # items.append(interface.MetadataItem(name='scene_token', array=scene_tokens, dtype=tf.string)) 
    return items

def nuscenes_mini_dill_metadata_producer(data):
    items = interface.MetadataList()
    sample_tokens = np.asarray([_.metadata['sample_token'] for _ in data])
    items.append(interface.MetadataItem(name='sample_token', array=sample_tokens, dtype=tf.string))
    # scene_tokens = np.asarray([_.metadata['real_scene_token'] for _ in data])
    # items.append(interface.MetadataItem(name='scene_token', array=scene_tokens, dtype=tf.string)) 
    return items

def carla_town01_A5_T20_metadata_producer(data):
    return interface.MetadataList()

def carla_town01_A1_T20_metadata_producer(data):
    return interface.MetadataList()

def carla_town01_A1_T20_lightstate_metadata_producer(data):
    return interface.MetadataList()

def default_metadata_producer(data):
    return interface.MetadataList()

PRODUCERS = {
    "trimodal_dataset": lambda *args, **kwargs: interface.MetadataList(),
    'nuscenes_shuffle_A5_dill': nuscenes_dill_metadata_producer,
    'carla_town01_A5_T20_json': carla_town01_A5_T20_metadata_producer,
    'carla_town01_A1_T20_json': carla_town01_A1_T20_metadata_producer,
    'carla_town01_A1_T20_lightstate_json': carla_town01_A1_T20_lightstate_metadata_producer,
    'carla_town01_A1_T20_lightstate_streamingloader_json': carla_town01_A1_T20_lightstate_metadata_producer,
    'carla_town01_A1_T20_v1_json': carla_town01_A1_T20_lightstate_metadata_producer,
    'carla_town01_A1_T20_v2_json': carla_town01_A1_T20_lightstate_metadata_producer,
    # Added
    'carla_town01_A5_T20_test_json': default_metadata_producer,
}
