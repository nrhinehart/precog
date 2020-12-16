
import atexit
import logging
import hydra
import functools
import pdb
import os
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
    def __init__(self, cfg, sess, dataset, writer, output_directory,
            epochs, evaluate_period, **kwargs):
        assert(isinstance(cfg.model.directory, str))
        
        log.info("Loading model...")
        ckpt, graph, self.tensor_collections = tfutil.load_annotated_model(
                cfg.model.directory, self.sess)
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.inference = interface.ESPInference(self.tensor_collections)
        self.input_singleton = self.inference.training_input
        
        log.info("Getting optimizer and other values")
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
        count = 0
        self.dataset.reset_split(split)
        
        while True:
            minibatch = self.dataset.get_minibatch(split=split,
                    input_singleton=self.input_singleton, is_training=False)
            if minibatch is None:
                break
            results.append(self.sess.run(self.evaluate_targets[split], minibatch))
            count += 1

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
        # throws error
        """
        tensorflow.python.framework.errors_impl.InvalidArgumentError: 2 root error(s) found.
          (0) Invalid argument: You must feed a value for placeholder tensor 'light_strings' with dtype string and shape [10]
        	 [[node light_strings (defined at /code/robotics/precog/precog/utils/tfutil.py:60) ]]
          (1) Invalid argument: You must feed a value for placeholder tensor 'light_strings' with dtype string and shape [10]
        	 [[node light_strings (defined at /code/robotics/precog/precog/utils/tfutil.py:60) ]]
        	 [[interpolate_bilinear_37/assert_equal/Assert/Assert/data_0/_5033]]

        """
        # summary = self.sess.run(self.split_summaries[split], fd)
        # self.writer.add_summary(summary, global_step=global_step)
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
        log.info("Training complete. Doing final save.")
        self.save()



@hydra.main(config_path='conf/esp_train_config.yaml')
def main(cfg):
    output_directory = os.path.realpath(os.getcwd())
    log.info("Output directory: {}".format(output_directory))
    log.info("Creating TensorFlow session...")
    sess = tfutil.create_session(
            allow_growth=cfg.hardware.allow_growth,
            per_process_gpu_memory_fraction=cfg.hardware.per_process_gpu_memory_fraction)
    writer = tf.compat.v1.summary.FileWriter(logdir=output_directory, flush_secs=5, max_queue=1)
    log.info("Loading the model...")
    dataset = hydra.utils.instantiate(cfg.dataset, **cfg.dataset.params)
    opt = SGDOptimizer(
            cfg=cfg,
            sess=sess,
            dataset=dataset,
            writer=writer,
            output_directory=output_directory,
            **cfg.optimizer.params)
    opt.optimize()

if __name__ == '__main__':
    main()
