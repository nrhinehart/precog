
import contextlib
import dill
import functools
import logging
import numpy as np
import os
import pdb
import tensorflow as tf

import precog.interface
import precog.plotting.plot as plot
import precog.utils.class_util as classu
import precog.utils.tensor_util as tensoru
import precog.utils.tfutil as tfutil
import precog.utils.log_util as logu
import precog.utils.esp_gsheets_util as espgsheetsu

log = logging.getLogger(os.path.basename(__file__))
tfv1 = tf.compat.v1

class SGDOptimizer(precog.interface.ESPOptimizer):
    @classu.member_initialize
    def __init__(self,
                 model_distribution,
                 data_distribution_proxy,
                 objective,
                 sample_metric,
                 dataset,
                 writer,
                 output_directory,
                 learning_rate,
                 cfg,
                 epochs=10000,
                 max_epoch_steps=1000000,
                 debug=False,
                 plot_period=500,
                 evaluate_period=500,
                 sess=None,
                 save_eager_figures=True,
                 record_to_sheets=True,
                 save_before_train=False,
                 plot_before_train=False,
                 plot_without_samples=False,
                 figsize=[8,20]):
        self._tf_optimizer = tfv1.train.RMSPropOptimizer(learning_rate)
        # Record collections for inputs and outputs.
        self.model_collections = tfutil.ModelCollections(
            names=['optimizing',
                    'apply_gradients',
                    'sample_input',
                    'infer_input',
                    'shared_input',
                    'sample_output', 'infer_output',
                    'sample_metric', 'infer_metric',
                    'intermediate_input', 'intermediate_label'])
        self.record_to_sheets = self.record_to_sheets and espgsheetsu.gsheetsu.have_pygsheets
        if self.record_to_sheets:
            self.results = espgsheetsu.ESPResults(
                tag=os.path.basename(self.output_directory),
                dataset=self.dataset.name,
                output_directory=self.output_directory)
        self.images_directory = os.path.join(self.output_directory, 'images')
        os.mkdir(self.images_directory)

    def optimize(self):
        
        ss = "Beginning ESP Optimization." + "\n\tDistribution: {}".format(self.model_distribution)
        ss += "\n\tObjective: {}".format(self.objective) + "\n\tDataset: {}".format(self.dataset)
        ss += "\n\tData distribution proxy: {}".format(self.data_distribution_proxy)
        log.info(ss)

        global_step = tfv1.train.get_or_create_global_step()
        is_eager = tf.executing_eagerly()
        
        if is_eager: assert(self.sess is None)
        else: assert(self.sess is not None)
        if not is_eager:
            self._static_create_singleton()
            self._static_prepare_samples(global_step)
        else:
            self.input_singleton = None
        
        if self.plot_before_train:
            minibatch = self.dataset.get_minibatch(
                    mb_idx=0, split='train',
                    input_singleton=self.input_singleton,
                    is_training=True)
            if is_eager: self._eager_plot(minibatch, global_step.numpy())
            else: self._static_plot(minibatch)

        # Prepare the model.
        if is_eager: self._eager_prepare()
        else: self._static_prepare_objective(global_step)

        if not is_eager and self.save_before_train:
            self.saver.save(self.sess,
                            os.path.join(self.output_directory, 'esp-model'),
                            write_meta_graph=True,
                            global_step=global_step)
        self._run_optimize(global_step, is_eager)

    def _run_optimize(self, global_step, is_eager, best_ehat_val=np.inf):
        log.info("Starting gradient descent")
        for epoch in range(self.epochs):
            # Reset the training data.
            self.dataset.reset_split('train')
            # Retrieve minibatches until the dataset runs out.
            while True:
                minibatch = self.dataset.get_minibatch(split='train',
                        input_singleton=self.input_singleton,
                        is_training=True)
                if minibatch is None: break
                # Step gradients.
                if is_eager: self._eager_gradient_step(minibatch, global_step, epoch)
                else: self._static_gradient_step(minibatch, global_step, epoch)
                # Possibly plot samples.
                if self._should_plot_now(global_step):
                    if is_eager: self._eager_plot(minibatch, global_step.numpy())
                    else: self._static_plot(minibatch)
                # Possibly evaluate splits.
                if self._should_evaluate_now(global_step):
                    # Evaluate val.
                    if is_eager: ehat_val = self._eager_evaluate_split('val', global_step)
                    else:
                        self.dataset.reset_split('val')
                        ehat_val = self._static_evaluate_split('val', global_step)
                    # Validation score is best. Update, and evaluate test.
                    if ehat_val < best_ehat_val:
                        best_ehat_val = ehat_val
                        if is_eager: ehat_val = self._eager_evaluate_split('test', global_step)
                        else:
                            self.dataset.reset_split('test')
                            ehat_val = self._static_evaluate_split('test', global_step)
                        if not is_eager:
                            self.saver.save(self.sess,
                                            os.path.join(self.output_directory, 'esp-model'),
                                            write_meta_graph=True,
                                            global_step=global_step)

    def _should_plot_now(self, global_step):
        is_eager = tf.executing_eagerly()
        if is_eager: return global_step.numpy() % self.plot_period == 0
        else: return self.sess.run(global_step) % self.plot_period == 0

    def _should_evaluate_now(self, global_step):
        is_eager = tf.executing_eagerly()
        if is_eager: return global_step.numpy() % self.evaluate_period == 0
        else: return self.sess.run(global_step) % self.evaluate_period == 0
        
    def _eager_prepare(self):
        self.input_singleton = None

    def _static_create_singleton(self):
        # Get a single minibatch outside the loop, because in static mode we'll need it to define more tensors.
        log.info("Fetching minibatch in order to create static-input placeholders")
        input_ = self.dataset.get_minibatch(split='train', mb_idx=0, is_training=True)
        self.input_singleton = input_.to_singleton()
        
    @logu.log_wrapi()
    def _static_prepare_samples(self, global_step):
        for t in self.input_singleton.sample_placeholders: self.model_collections.add_sample_input(t)
        for t in self.input_singleton.placeholders: self.model_collections.add_infer_input(t)

        # Track some intermediate tensors that we'll want to reconstruct when we load the model.
        self.model_collections.add_intermediate_input(self.input_singleton.phi.S_past_car_frames)
        self.model_collections.add_intermediate_input(self.input_singleton.phi.S_past_grid_frame)
        self.model_collections.add_intermediate_input(self.input_singleton.phi_m.agent_counts)
        self.model_collections.add_intermediate_label(self.input_singleton.experts.S_future_car_frames)
        self.model_collections.add_intermediate_label(self.input_singleton.experts.S_future_grid_frame)
        
        log.info("Computing static-mode samples.")
        self.sampled_output = self.model_distribution.sample(
            phi=self.input_singleton.phi, phi_metadata=self.input_singleton.phi_m, T=self.dataset.T)
        # dbg
        self.model_distribution.bijection.check_gradients(self.sampled_output.base_and_log_q.Z_sample)
        # Z input is a sample input
        self.model_collections.add_sample_input(self.sampled_output.base_and_log_q.Z_sample)
        
        # Record outputs for sampling.
        for t in self.sampled_output.rollout.rollout_outputs: self.model_collections.add_sample_output(t)
        # Log q is a sample output.
        self.model_collections.add_sample_output(self.sampled_output.base_and_log_q.log_q_samples)

        # Functions to write a numpy image into tensorboard. The function takes a single input (np.ndarray)
        self.partial_np2tb_smbs = []
        for c in range(tensoru.size(self.input_singleton.phi.overhead_features, -1)):
            self.partial_np2tb_smbs.append(plot.bind_write_np_image_to_tb(
                sess=self.sess, writer=self.writer, global_step=global_step, key='sampled_minibatch_bev_{}'.format(c)))
        self.partial_np2tb_smb_joint = plot.bind_write_np_image_to_tb(
            sess=self.sess, writer=self.writer, global_step=global_step, key='sampled_minibatch_bev_joint')
        self.partial_np2tb_cnn0 = plot.bind_write_np_image_to_tb(
            sess=self.sess, writer=self.writer, global_step=global_step, key='CNN_0')        
        
        log.info("Initializing variables...")
        self.sess.run(tfv1.global_variables_initializer())

    def _static_prepare_objective(self, global_step):

        # Instantiate the training minimization criterion, some other stuff.
        log.info("Computing static-mode objective.")
        self.criterion, self.Hpq, self.ehat, self.expert_roll, self.Hqphat = self.objective(
            self.model_distribution, self.sampled_output,
            self.data_distribution_proxy, self.input_singleton).unpack()

        # map_checks0 = tfutil.ensure_no_crossbatch_gradients(self.sampled_output.rollout.S_car_frames,
        #                                                     self.input_singleton.phi.overhead_features)
        # map_checks = tfutil.assert_no_crossbatch_gradients(self.sampled_output.rollout.S_world_frame,
        #                                                    self.input_singleton.phi.overhead_features,
        #                                                    n_max=2)

        all_checks = []
        if self.debug:
            # Ensure that the past positions can't affect the future positions of other items in the minibatch.
            past_checks = tfutil.assert_no_crossbatch_gradients(
                self.sampled_output.rollout.S_world_frame,
                self.input_singleton.phi.S_past_car_frames,
                n_max=5,
                name_out="S_world_frame",
                name_in="S_past_car_frames")
            
            all_checks += past_checks
            # light_checks = tfutil.assert_no_crossbatch_gradients(self.sampled_output.rollout.S_world_frame,
            #                                                      self.input_singleton.phi.light_features,
            #                                                      n_max=5,
            #                                                      name_out="S_world_frame",
            #                                                      name_in="light_features")
            
            # all_checks += light_checks

        all_checks += self.model_distribution.bijection.crossbatch_asserts
            
        self.sample_metric_tensor = self.sample_metric(self.sampled_output, self.input_singleton.experts)
        # Record the log prob output for inference.
        self.model_collections.add_infer_output(tf.identity(-1. * self.Hpq, 'log_q_expert'))

        # Create scalar summaries.
        self.split_summaries = {}
        self.evaluate_targets = {}
        self.mean_Hpq = tf.reduce_mean(self.Hpq, name='mean_Hpq')
        self.mean_ehat = tf.reduce_mean(self.ehat, name='mean_ehat')
        self.mean_Hqphat = tf.reduce_mean(self.Hqphat, name='mean_Hqphat')
        self.mean_sample_metric = tf.reduce_mean(self.sample_metric_tensor, name='mean_{}'.format(self.sample_metric))

        self.model_collections.add_optimizing(
                tf.identity(self.Hpq, name='Hpq'))
        self.model_collections.add_optimizing(
                tf.identity(self.criterion, name='criterion'))
        self.model_collections.add_optimizing(self.ehat)
        self.model_collections.add_optimizing(
                tf.identity(self.Hqphat, name='Hqphat'))
        self.model_collections.add_optimizing(
                tf.identity(self.sample_metric_tensor, name='sample_metric_tensor'))
        self.model_collections.add_optimizing(
                tf.identity(self.mean_sample_metric, name='mean_sample_metric'))

        # Record inference and sample metrics.
        for tensor in [self.mean_Hpq, self.mean_ehat, self.Hpq, self.ehat]:
            self.model_collections.add_infer_metric(tensor)
        for tensor in [self.mean_sample_metric, self.mean_Hqphat, self.sample_metric_tensor]:
            self.model_collections.add_sample_metric(tensor)

        # Find the inputs that both sample and input everything requires.
        sample_input = set(self.model_collections.tensor_collections['sample_input'])
        infer_input = set(self.model_collections.tensor_collections['infer_input'])
        shared_input = sample_input & infer_input
        for tensor in shared_input: self.model_collections.add_shared_input(tensor)

        # Ensure that the model's variables fully-parameterize all of the trainable variables.
        tfutil.require_complete_parameterization(self.model_distribution.variables)

        # Instantiate the gradients now.
        log.info("Computing static-mode gradients...")
        dmin_criterion_dvars, vars_ = zip(*self._tf_optimizer.compute_gradients(
            self.criterion, self.model_distribution.variables))
        
        # Ensure all gradients exist.
        assert(None not in dmin_criterion_dvars)
        
        # Gradient norm clip.
        dmin_criterion_dvars = zip(tf.clip_by_global_norm(dmin_criterion_dvars, 5.)[0], vars_)
        
        # Treat all_checks as ordered dependencies, so earlier ones run first.
        managers = [tf.control_dependencies([check]) for check in all_checks]
        with contextlib.ExitStack() as stack:
            for manager in managers:
                stack.enter_context(manager)
            apply_gradients = self._tf_optimizer.apply_gradients(dmin_criterion_dvars, global_step=global_step, name='apply_gradients')
        self.model_collections.add_apply_gradients(apply_gradients)
        # with tf.control_dependencies(all_checks):
        # Create separate summary ops for each split.
        
        for s in ('train', 'val', 'test'):
            summaries = []
            summaries.append(tfv1.summary.scalar("{}/mean_Hpq".format(s), self.mean_Hpq))
            summaries.append(tfv1.summary.scalar("{}/mean_ehat".format(s), self.mean_ehat))
            summaries.append(tfv1.summary.scalar("{}/mean_Hqphat".format(s), self.mean_Hqphat))
            summaries.append(tfv1.summary.scalar("{}/mean_{}".format(s, self.sample_metric), self.mean_sample_metric))
            self.split_summaries[s] = tfv1.summary.merge(summaries, name=f"{s}_summaries")
            self.model_collections.add_optimizing(self.split_summaries[s])
            self.evaluate_targets[s] = [self.Hpq, self.ehat, self.Hqphat, self.sample_metric_tensor]
                    
        # Operations / Tensors to run for each gradient descent step.
        self.gradient_step_targets = [
                global_step, apply_gradients, self.Hpq, self.ehat,
                self.split_summaries['train']]

        # Re-init for the optimizer.
        self.sess.run(tfv1.global_variables_initializer())        

        log.info("Saving model")
        self.saver = tfv1.train.Saver(var_list=tfv1.get_collection(tfv1.GraphKeys.GLOBAL_VARIABLES))
        self.saver.save(self.sess, os.path.join(self.output_directory, 'esp-model'), write_meta_graph=True, global_step=global_step)
        self.sess.graph.finalize()
        
        for name in self.model_collections.names: assert(len(self.model_collections.tensor_collections[name]) > 0)
        
        # Save the names of the collections so that we can easily retrieve them once the model is loaded.
        with open(self.output_directory + '/collections.dill', 'wb') as f:
            dill.dump(self.model_collections.names, f)
            
    def _eager_gradient_step(self, minibatch, global_step, epoch):
        # Compute gradients on-the-fly.
        with tf.GradientTape() as tape:
            if self.objective.requires_samples or self.sample_metric is not None: 
                sampled_output = self.model_distribution.sample(phi=minibatch.phi, phi_metadata=minibatch.phi_m, T=self.dataset.T)
            else:
                sampled_output = None

            self.obj_return = self.objective(self.model_distribution, sampled_output, self.data_distribution_proxy, minibatch) 
            criterion, Hpq, ehat, roll, Hqphat = self.obj_return.unpack()

            # TODO clip these gradients.
            dmin_criterion_dvars = tape.gradient(criterion, self.model_distribution.variables)
            
        # Apply the gradients to the variables.
        self._tf_optimizer.apply_gradients(zip(dmin_criterion_dvars, self.model_distribution.variables), global_step)
        
        # Possibly debug the bijection.
        if self.debug:
            self.model_distribution.bijection.eager_ensure_forward_bijection(minibatch.experts.S_future_car_frames, phi=minibatch.phi)
            
        # Log info.
        log.info("global step {:06d}, epoch {:06d}. H(p,q)={:7.2f}, ehat={:4.2f}, H(q,phat)={:6.2f}".format(
            global_step.numpy(), epoch, np.mean(Hpq), np.mean(ehat), np.mean(Hqphat)))
        
        if np.any(ehat < 0):
            log.error("Ehat is negative! Did you fail to perturb your data?")

        self.sample_metric_tensor = self.sample_metric(sampled_output, minibatch.experts)
            
        # Log tensorboard scalars.
        tf.contrib.summary.scalar("train/mean_Hpq", np.mean(Hpq))
        tf.contrib.summary.scalar("train/mean_ehat", np.mean(ehat))
        if np.isfinite(Hqphat).all(): tf.contrib.summary.scalar("train/mean_Hqphat", np.mean(Hqphat))

        return np.mean(ehat)

    def _static_gradient_step(self, minibatch, global_step, epoch):
        # Static mode gradient update.
        global_step, _, Hpq, ehat, summary = self.sess.run(self.gradient_step_targets, minibatch)
        log.info("global step {:08d}, epoch {:08d}. H(p,q)={:03.2f}, ehat={:.2f}".format(
            global_step, epoch, np.mean(Hpq), np.mean(ehat)))
        
        # Log tensorboard scalars.
        self.writer.add_summary(summary, global_step=global_step)
        return np.mean(ehat)

    @logu.log_wrapi()
    def _eager_plot(self, minibatch, global_step):
        sampled_output = self.model_distribution.sample(phi=minibatch.phi, phi_metadata=minibatch.phi_m, T=self.dataset.T)
        #     fig, axes = plot.plot_coordinate_frames(self.obj_return.rollout.rollout_car_frames_list_grid)
        fig, axes = plot.get_figure()
        plot.plot_sampled_minibatch(
            sampled_output,
            experts=minibatch.experts,
            figsize=self.figsize,
            without_samples=self.plot_without_samples,
            fig=fig,
            axes=axes)
        if self.save_eager_figures: fig.savefig(self.images_directory + '/{:08d}.tiff'.format(global_step))
        plot.figsclose()

    @logu.log_wrapi()        
    def _static_plot(self, minibatch):
        log.debug("Static plotting")
        sessrun = functools.partial(self.sess.run, feed_dict=minibatch)
        # Convert data to numpy to prepare for plotting.
        sampled_output_np = self.sampled_output.to_numpy(sessrun)
        experts_np = self.input_singleton.experts.to_numpy(sessrun)

        # Plot things over every channel of the BEV.
        for c in range(tensoru.size(self.input_singleton.phi.overhead_features, -1)):
            plot_bev_kwargs = {'onechannel': True, 'channel_idx': c, 'allchannel': False}
            plot.plot_sampled_minibatch(
                sampled_output=sampled_output_np,
                experts=experts_np,
                partial_write_np_image_to_tb=self.partial_np2tb_smbs[c],
                figsize=self.figsize,
                without_samples=self.plot_without_samples,
                plot_bev_kwargs=plot_bev_kwargs,
                tensorstr='_bev-{}'.format(c))

        plot_bev_kwargs = {'onechannel': False, 'allchannel': False}
        plot.plot_sampled_minibatch(
            sampled_output=sampled_output_np,
            experts=experts_np,
            partial_write_np_image_to_tb=self.partial_np2tb_smb_joint,
            figsize=self.figsize,
            without_samples=self.plot_without_samples,
            plot_bev_kwargs=plot_bev_kwargs,
            tensorstr='_bev-joint')

        try:
            feature_map = sessrun(self.model_distribution.bijection.feature_map)        
            feature_map_fig_w = int(np.ceil(np.sqrt(feature_map.shape[-1])))
            plot.plot_feature_map(
                feature_map=feature_map, partial_write_np_image_to_tb=self.partial_np2tb_cnn0, nrows=feature_map_fig_w, ncols=feature_map_fig_w)
        except AttributeError as e:
            log.error(e)

    @logu.log_wrapi()                
    def _static_evaluate_split(self, split, global_step):
        results = []
        count = 0

        # Ensure we start at the beginning of the split.
        self.dataset.reset_split(split)
        
        while True:
            # Debug evaluate with small amount of data.
            if self.debug and count > 10: break
            minibatch = self.dataset.get_minibatch(split=split, input_singleton=self.input_singleton, is_training=False)
            if minibatch is None: break
            results.append(self.sess.run(self.evaluate_targets[split], minibatch))
            count += 1

        mean_Hpqs, mean_ehats, mean_Hqphats, mean_sample_metrics = map(np.mean, zip(*results))
        global_step = self.sess.run(global_step)

        fd = {self.mean_ehat: mean_ehats, self.mean_Hpq: mean_Hpqs, self.mean_Hqphat: mean_Hqphats, self.mean_sample_metric: mean_sample_metrics}
        strs = [('ehat', mean_ehats), (str(self.sample_metric), mean_sample_metrics)]
        
        # Create the summary by injecting manually-computed results.
        summary = self.sess.run(self.split_summaries[split], fd)
        self.writer.add_summary(summary, global_step=global_step)
        if self.record_to_sheets: self.results.update(split, global_step, strs)
        return mean_ehats

    @logu.log_wrapi()                
    def _eager_evaluate_split(self, split, global_step):
        results = []
        sample_metric_results = []
        while True:
            minibatch = self.dataset.get_minibatch(split=split, input_singleton=self.input_singleton, is_training=False)
            if minibatch is None: break
            sampled_output = self.model_distribution.sample(phi=minibatch.phi, phi_metadata=minibatch.phi_m, T=self.dataset.T)
            sample_metric_results.append(self.sample_metric(sampled_output, minibatch.experts))
            results.append(self.objective(
                self.model_distribution, sampled_output, self.data_distribution_proxy, minibatch))
        _, mean_Hpqs, mean_ehats, mean_Hqphats = map(np.mean, zip(*results))
        mean_sample_metric = np.mean(sample_metric_results)

        # Log tensorboard scalars.
        tf.contrib.summary.scalar("{}/mean_Hpq".format(split), mean_Hpqs, step=global_step.numpy())
        tf.contrib.summary.scalar("{}/mean_Hqphat".format(split), mean_Hqphats, step=global_step.numpy())
        tf.contrib.summary.scalar("{}/mean_ehat".format(split), mean_ehats, step=global_step.numpy())
        tf.contrib.summary.scalar("{}/mean_{}".format(split, self.sample_metric), mean_sample_metric, step=global_step.numpy())
