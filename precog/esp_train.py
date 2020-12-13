
import atexit
import hydra
import logging
import numpy as np
import os
import pdb
import sys
import tensorflow as tf

import precog.distribution.bijective_distribution as bijective_distribution
import precog.optimizer.sgd_optimizer as sgd_optimizer

import precog.utils.log_util as logu
import precog.utils.tfutil as tfutil
import precog.utils.rand_util as randu

log = logging.getLogger(__file__)

np.set_printoptions(precision=10, suppress=True)

@hydra.main(config_path='conf/esp_train_config.yaml')
def main(cfg):
    optimize(prepare_for_optimization(cfg))

def optimize(opt):
    # Fit the distribution according to the objective on the dataset.
    with tf.contrib.summary.always_record_summaries(): opt.optimize()

def prepare_for_optimization(cfg):
    # Log info about the program state.
    print("Starting main. argv: {}".format(' '.join(sys.argv)))
    log.info("Starting main. argv: {}".format(' '.join(sys.argv)))
    log.info("\n\nConfig:\n===\n{}".format(cfg.pretty()))
    log.info("Sha: {}. Dirty: {}".format(*logu.get_sha_and_dirty()))
    # Grab a reference to the output directory (automatically created).
    output_directory = os.path.realpath(os.getcwd())    
    log.info("Output directory: {}".format(output_directory))
    # Seed the RNG.
    randu.seed(cfg.main.seed)
    # Create session.
    sess = instantiate_session(cfg)
    # Create writer.
    writer = instantiate_writer(cfg, output_directory)
    # Register cleanup.
    atexit.register(query_purge_directory_and_writer, output_directory, writer)
    # Create optimizer.
    opt = instantiate_optimizer(cfg, writer=writer, output_directory=output_directory, sess=sess)
    return opt

def instantiate_session(cfg):
    if cfg.main.eager:
        log.info("Running in eager mode")
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = cfg.hardware.per_process_gpu_memory_fraction
        # config.gpu_options.allow_growth = True
        # If for some reason we've created a graph already, trash it to enable eager executation.
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_eager_execution(config=config)
        sess = None
    else:
        log.info("Running in static mode")
        # Create the session now because it spits out lots of clutter.
        sess = tfutil.create_session(
            allow_growth=cfg.hardware.allow_growth, per_process_gpu_memory_fraction=cfg.hardware.per_process_gpu_memory_fraction)
    return sess

def instantiate_writer(cfg, output_directory):
    # Intantiate the tensorboard writer.
    if cfg.main.eager:
        writer = tf.contrib.summary.create_file_writer(logdir=output_directory, flush_millis=5000, max_queue=1)
        writer.set_as_default()
    else:
        writer = tf.compat.v1.summary.FileWriter(logdir=output_directory, flush_secs=5, max_queue=1)
    return writer

def instantiate_optimizer(cfg, writer, output_directory, sess):
    # Instantiate the parameterized bijection.
    bijection = hydra.utils.instantiate(cfg.bijection)

    # Instantiate the distribution with the parameterized bijection.
    model_distribution = bijective_distribution.ESPBijectiveDistribution(
        bijection=bijection, **cfg.distribution.params)

    # Instantiate the proxy distribution for potential reverse-CE minimization.
    data_distribution_proxy = hydra.utils.instantiate(cfg.proxy)

    # Instantiate the objective.
    objective = hydra.utils.instantiate(cfg.objective)

    # Instantiate the sample metric.
    sample_metric = hydra.utils.instantiate(cfg.sample_metric)

    # Instantiate the dataset.
    dataset = hydra.utils.instantiate(cfg.dataset, **cfg.dataset.params)

    assert dataset.max_A >= bijection.A, "Dataset's A is less than bijection's A!"
    
    # Instantiate the optimizer
    opt = sgd_optimizer.SGDOptimizer(model_distribution=model_distribution,
                                     data_distribution_proxy=data_distribution_proxy,
                                     objective=objective,
                                     sample_metric=sample_metric,
                                     dataset=dataset,
                                     writer=writer,
                                     output_directory=output_directory,
                                     sess=sess,
                                     cfg=cfg,
                                     **cfg.optimizer.params)
    return opt

def query_purge_directory_and_writer(directory, writer):
    """

    :param directory: directory to purge
    :param writer: pass the FileWriter so we can properly close it.
    :returns: 
    :rtype: 

    """
    writer.close()
    import utils.log_util as logu
    logu.query_purge_directory(directory)

if __name__ == '__main__':
    main()
