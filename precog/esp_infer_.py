
import atexit
import logging
import hydra
import functools
import pdb
import numpy as np
import os
import scipy.stats
import skimage.io

import precog.utils.log_util as logu
import precog.utils.tfutil as tfutil
import precog.interface as interface
import precog.plotting.plot as plot
import precog.utils.class_util as classu

log = logging.getLogger(os.path.basename(__file__))

class SGDOptimizer():
    def __init__(self, cfg, objective, debug=False):
        self.objective = objective
        self.debug = debug
        ckpt, graph, self.tensor_collections = tfutil.load_annotated_model(
                cfg.model.directory, sess)

    def optimize(self):
        tfv1 = tf.compat.v1


        # global_step = tf.compat.v1.train.get_or_create_global_step()
        # assert(not tf.executing_eagerly())
        # self._static_create_singleton()
        # self._static_prepare_samples(global_step)
        # self._static_prepare_objective(global_step)


@hydra.main(config_path='conf/esp_infer_config.yaml')
def main(cfg):
    log.info("Creating TensorFlow session...")
    sess = tfutil.create_session(
            allow_growth=cfg.hardware.allow_growth,
            per_process_gpu_memory_fraction=cfg.hardware.per_process_gpu_memory_fraction)
    writer = tf.compat.v1.summary.FileWriter(logdir=output_directory, flush_secs=5, max_queue=1)
    log.info("Loading the model...")
    ckpt, graph, tensor_collections = tfutil.load_annotated_model(cfg.model.directory, sess)
    inference = interface.ESPInference(tensor_collections)

    cfg.dataset.params.T = inference.metadata.T
    cfg.dataset.params.B = inference.metadata.B
    dataset = hydra.utils.instantiate(cfg.dataset, **cfg.dataset.params)


@hydra.main(config_path='conf/esp_infer_config.yaml')
def old_main(cfg):
    assert(cfg.main.plot or cfg.main.compute_metrics)
    output_directory = os.path.realpath(os.getcwd())
    images_directory = os.path.join(output_directory, 'images')
    os.mkdir(images_directory)
    log.info("\n\nConfig:\n===\n{}".format(cfg.pretty()))

    atexit.register(logu.query_purge_directory, output_directory)

    # Instantiate the session.
    sess = tfutil.create_session(
            allow_growth=cfg.hardware.allow_growth,
            per_process_gpu_memory_fraction=cfg.hardware.per_process_gpu_memory_fraction)

    # Load the model and the tensor collections.
    log.info("Loading the model...")
    ckpt, graph, tensor_collections = tfutil.load_annotated_model(cfg.model.directory, sess)
    inference = interface.ESPInference(tensor_collections)
    sample_metrics = tfutil.get_collection_dict(tensor_collections['sample_metric'])
    if cfg.main.compute_metrics:
        infer_metrics = tfutil.get_collection_dict(tensor_collections['infer_metric'])
        metrics = {**infer_metrics, **sample_metrics}
        all_metrics = {_: [] for _ in metrics.keys()}

    # Instantiate the dataset.
    cfg.dataset.params.T = inference.metadata.T
    cfg.dataset.params.B = inference.metadata.B
    dataset = hydra.utils.instantiate(cfg.dataset, **cfg.dataset.params)

    log.info("Beginning evaluation. Model: {}".format(ckpt))    
    count = 0
    while True:
        minibatch = dataset.get_minibatch(split=cfg.split,
                input_singleton=inference.training_input, is_training=False)
        if not cfg.main.compute_metrics:
            for t in inference.training_input.experts.tensors:
                try:
                    del minibatch[t]
                except KeyError:
                    pass
        if minibatch is None: break
        sessrun = functools.partial(sess.run, feed_dict=minibatch)
        try:
            # Run sampling and convert to numpy.
            sampled_output_np = inference.sampled_output.to_numpy(sessrun)
            
            if cfg.main.compute_metrics:
                # Get experts in numpy version.
                experts_np = inference.training_input.experts.to_numpy(sessrun)
                # Compute and store metrics.
                metrics_results = dict(zip(metrics.keys(), sessrun(list(metrics.values()))))            
                for k, val in metrics_results.items(): all_metrics[k].append(val)
            else:
                experts_np = None
        except ValueError as v:
            print("Got value error: '{}'\n Are you sure the provided dataset ('{}') is compatible with the model?".format(
                v, cfg.dataset))
            raise v
        if cfg.main.plot:
            log.info("Plotting...")
            # for b in range(10):
            for b in range(inference.metadata.B):
                im = plot.plot_sample(sampled_output_np,
                        experts_np, b=b,
                        partial_write_np_image_to_tb=lambda x:x,
                        figsize=cfg.images.figsize,
                        bev_kwargs=cfg.plotting.bev_kwargs)
                skimage.io.imsave('{}/esp_samples_{:05d}.{}'.format(images_directory, count, cfg.images.ext), im[0,...,:3])
                log.info("Plotted.")
                count += 1
        if cfg.main.compute_metrics:
            for k, vals in all_metrics.items():
                log.info("Mean,sem '{}'={:.3f} +- {:.3f}".format(k, np.mean(vals), scipy.stats.sem(vals, axis=None)))
    
    if cfg.main.compute_metrics:
        log.info("Final metrics\n=====\n")
        for k, vals in all_metrics.items():
            log.info("Mean,sem '{}'={:.3f} +- {:.3f}".format(k, np.mean(vals), scipy.stats.sem(vals, axis=None)))
    
if __name__ == '__main__':
    main()
