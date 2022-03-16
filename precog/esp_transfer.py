
import atexit
import logging
import functools
import os
import os.path
import shutil
import pdb
import hydra
import numpy as np
import tensorflow as tf

import precog.utils.log_util as logu
import precog.utils.tfutil as tfutil
import precog.interface as interface
import precog.plotting.plot as plot
import precog.utils.class_util as classu

tfv1 = tf.compat.v1
log = logging.getLogger(os.path.basename(__file__))

class SGDOptimizer():
    def _add_props(self, coll):
        for t in coll:
            if '/' in t.name:
                name, *_ = t.name.split('/')
            elif ':' in t.name:
                name, *_ = t.name.split(':')
            else:
                name = t.name
            if not hasattr(self, name):
                setattr(self, name, t)

    @classu.member_initialize
    def __init__(self, cfg, sess, dataset, inference,
            tensor_collections, output_directory,
            epochs, evaluate_period, **kwargs):
        
        self.writer = tf.compat.v1.summary.FileWriter(
            logdir=output_directory, flush_secs=5, max_queue=1)
        
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        
        self.input_singleton = self.inference.training_input
        
        log.info("Retrieving tensors...")
        self._add_props(self.tensor_collections['optimizing'])
        self._add_props(self.tensor_collections['apply_gradients'])
        self._add_props(self.tensor_collections['infer_metric'])
        self._add_props(self.tensor_collections['sample_metric'])

        self.split_summaries = {}
        self.evaluate_targets = {}
        for s in ('train', 'val', 'test'):
            summaries = getattr(self, f"{s}_summaries")
            self.split_summaries[s] = summaries
            self.evaluate_targets[s] = [
                    self.Hpq, self.ehat, self.Hqphat,
                    self.sample_metric_tensor]
        self.gradient_step_targets = [
                self.global_step, self.apply_gradients, self.Hpq, self.ehat,
                self.split_summaries['train']]
        
        log.info("Setting up model saving...")
        var_list = tfv1.get_collection(tfv1.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tfv1.train.Saver(var_list=var_list)

    def save(self):
        self.saver.save(
                self.sess,
                os.path.join(self.output_directory, 'esp-model'),
                write_meta_graph=True,
                global_step=self.global_step)

    def _static_evaluate_split(self, split):
        log.info(f"Evaluating split {split}")
        results = []
        self.dataset.reset_split(split)
        
        while True:
            minibatch = self.dataset.get_minibatch(split=split,
                    input_singleton=self.input_singleton, is_training=False)
            if minibatch is None:
                break
            results.append(self.sess.run(self.evaluate_targets[split], minibatch))

        mean_Hpqs, mean_ehats, mean_Hqphats, mean_sample_metrics = map(np.mean, zip(*results))
        global_step = self.sess.run(self.global_step)
        fd = {
                self.mean_ehat: mean_ehats,
                self.mean_Hpq: mean_Hpqs,
                self.mean_Hqphat: mean_Hqphats,
                self.mean_sample_metric: mean_sample_metrics}
        for t, v in fd.items():
            name, _ = t.name.split(':')
            log.info("Evaluate {} {}={:.3f}".format(
                    split, name, v))

        # need to pass minibatch in feed dict to summaries for some reason.
        self.dataset.reset_split(split)
        minibatch = self.dataset.get_minibatch(split=split,
                    input_singleton=self.input_singleton, is_training=False)
        fd.update(minibatch)
        summary = self.sess.run(self.split_summaries[split], fd)
        self.writer.add_summary(summary, global_step=global_step)

        return mean_ehats

    def _should_evaluate_now(self):
        return self.sess.run(self.global_step) % self.evaluate_period == 0

    def _static_gradient_step(self, minibatch, epoch):
        global_step, _, Hpq, ehat, summary = self.sess.run(self.gradient_step_targets, minibatch)
        log.info("Global step {:08d}, epoch {:08d}. H(p,q)={:.3f}, ehat={:.3f}".format(
                global_step, epoch, np.mean(Hpq), np.mean(ehat)))
        self.writer.add_summary(summary, global_step=global_step)
        return np.mean(ehat)

    def optimize(self):
        assert(not tf.executing_eagerly())
        best_ehat_val = self._static_evaluate_split('val')
        for epoch in range(self.epochs):
            self.dataset.reset_split('train')
            while True:
                minibatch = self.dataset.get_minibatch(split='train',
                        input_singleton=self.input_singleton,
                        is_training=True)
                if minibatch is None:
                    break
                self._static_gradient_step(minibatch, epoch)
                if self._should_evaluate_now():
                    ehat_val = self._static_evaluate_split('val')
                    log.info("Validation ehat {:.3f}, best ehat {:.3f}".format(
                            ehat_val, best_ehat_val))
                    if ehat_val < best_ehat_val:
                        global_step = self.sess.run(self.global_step)
                        log.info(f"Saving model at global step {global_step}")
                        best_ehat_val = ehat_val
                        self.save()
        log.info("Training complete.")


@hydra.main(config_path='conf/esp_transfer.yaml')
def main(cfg):
    log.info("Running transfer learning.")
    output_directory = os.path.realpath(os.getcwd())
    atexit.register(logu.query_purge_directory, output_directory)
    log.info("Output directory: {}".format(output_directory))

    log.info("Creating TensorFlow session...")
    sess = tfutil.create_session(
            allow_growth=cfg.hardware.allow_growth,
            per_process_gpu_memory_fraction=cfg.hardware.per_process_gpu_memory_fraction)

    log.info("Loading model...")
    assert(os.path.isdir(cfg.model.directory))
    ckpt, graph, tensor_collections = tfutil.load_annotated_model(
            cfg.model.directory, sess)
    inference = interface.ESPInference(tensor_collections)
    shutil.copy2(
            os.path.join(cfg.model.directory, 'collections.dill'),
            os.path.join(output_directory, 'collections.dill'))
    
    log.info("Loading dataset...")
    cfg.dataset.params.T = inference.metadata.T
    cfg.dataset.params.B = inference.metadata.B
    cfg.dataset.params.A = inference.metadata.A
    cfg.dataset.params.W = inference.phi_metadata.H
    dataset = hydra.utils.instantiate(cfg.dataset, **cfg.dataset.params)

    log.info("\n\nConfig:\n===\n{}".format(cfg.pretty()))

    log.info("Loading the SGD Optimizer...")
    opt = SGDOptimizer(
            cfg=cfg,
            sess=sess,
            dataset=dataset,
            inference=inference,
            tensor_collections=tensor_collections,
            output_directory=output_directory,
            **cfg.optimizer.params)
    log.info("Starting optimization...")
    opt.optimize()
    atexit.unregister(logu.query_purge_directory)

if __name__ == '__main__':
    main()
