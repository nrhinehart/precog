
import functools
import logging
import operator
import pdb

import numpy as np
import tensorflow as tf

import precog.interface as interface
import precog.bijection.esp_bijection as esp_bijection
import precog.bijection.bijection_helpers as bijection_helpers

import precog.utils.tfutil as tfutil
import precog.utils.tensor_util as tensoru
import precog.utils.log_util as logu
import precog.utils.class_util as classu

log = logging.getLogger(__file__)

class SocialConvRNN(esp_bijection.ESPJointTrajectoryBijectionMixin, interface.ESPJointTrajectoryBijection):
    @logu.log_wrapi()
    @classu.member_initialize
    def __init__(self,
                 A,
                 debug_static,
                 debug_eager,
                 mlpconf,
                 rnnconf,
                 whiskerconf,
                 cnnconf,
                 socialconf,
                 lightconf):
        """
        Create a CNN-RNN to predict trajectories, with social features per-timestep

        :param A: 
        :param mlpconf: see social_convrnn.yaml
        :param rnnconf: see social_convrnn.yaml
        :param whiskerconf: see social_convrnn.yaml
        :param cnnconf: see social_convrnn.yaml
        :param socialconf: see social_convrnn.yaml
        :param lightconf: see social_convrnn.yaml
        :returns: 
        :rtype: 

        """
        is_eager = tf.executing_eagerly()
        if debug_eager: assert(is_eager)
        if debug_static: assert(not is_eager)
        
        self.past_rnn = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=rnnconf.past_gru_units)

        self.past_encodings = None
        self.feature_map = None
        self.whiskers_grid = None

        # For debugging
        self.crossbatch_asserts = []
        
        self.G = 50
        # TODO Hardcoded D!
        self.D = 2

        # Layers for convnet.
        Conv2D = tf.compat.v1.layers.Conv2D
        Dense = tf.compat.v1.layers.Dense
        # Activations.
        conv_act = getattr(tf.nn, cnnconf.activation)
        mlp_act = getattr(tf.nn, mlpconf.activation)

        self.convnet = []
        if self.cnnconf.do_batchnorm:
            # TEMP
            assert(self.cnnconf.create_residual_connections)
            self.batchnorms = []
        for i in range(cnnconf.n_conv_layers):        
            self.convnet.append(Conv2D(filters=cnnconf.conv_filters, kernel_size=cnnconf.kernel_size, activation=conv_act, padding='same'))
            if self.cnnconf.do_batchnorm: self.batchnorms.append(tf.compat.v1.keras.layers.BatchNormalization(fused=False))
            
        self.convnet.append(Conv2D(filters=cnnconf.F, kernel_size=cnnconf.kernel_size, activation=conv_act, padding='same'))
        if self.cnnconf.do_batchnorm: self.batchnorms.append(tf.compat.v1.layers.BatchNormalization(fused=False))

        # MLP of social features.
        if self.socialconf.use_social_feats:
            self.social_mlp = [Dense(mlpconf.mlp_units, activation=mlp_act), tf.compat.v1.layers.Dense(self.G)]
        if self.mlpconf.do_prernn_mlp:
            self.prernn_mlp = [Dense(mlpconf.mlp_units, activation=mlp_act) for _ in range(self.mlpconf.n_prernn_layers)]

        # Post-GRU MLP. 
        self.mlp = [Dense(mlpconf.mlp_units, activation=mlp_act) for _ in range(self.mlpconf.n_postrnn_layers)]
        self.mlp += [Dense(self.D ** 2 + self.D)]
        
        self.rnn = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=rnnconf.future_gru_units)
        whiskers = bijection_helpers.generate_whisker_template(
            radii=whiskerconf.radii, arclength=whiskerconf.arclength, n_samples=whiskerconf.n_samples)
        self.whisker_template = tf.convert_to_tensor(whiskers)[None, None]
        self.step_generate_record = []
        
    @property
    def variables(self):
        vars_ = ([]
                 + tensoru.layer_variables(self.mlp)
                 + self.past_rnn.variables)
        if self.rnnconf.use_future_rnn:
            vars_ += self.rnn.variables
        if self.socialconf.use_social_map_feats:
            vars_ += tensoru.layer_variables(self.convnet)
        if self.socialconf.use_social_feats:
            vars_ += tensoru.layer_variables(self.social_mlp)
        if self.rnnconf.past_do_preconv:
            vars_ += [self.preconv_W, self.preconv_b]
        if self.mlpconf.do_prernn_mlp:
            vars_ += tensoru.layer_variables(self.prernn_mlp)
        return vars_

    @property
    def current_metadata(self):
        return {
            'whiskers_grid': self.whiskers_grid,
            'local2grid': self.current_local2grid,
            'local2world': self.current_local2world
        }

    def _prepare(self, batch_shape, phi):
        self.t = 0
        ldb = lambda x: log.info("Step {}, ".format(self.t) + x)
        self.step_generate_record.append({})
        assert(isinstance(batch_shape, tuple))
        # (B, ...), e.g. (B,); (B,K), etc.
        self.batch_shape = batch_shape
        self.batch_size = functools.reduce(operator.mul, self.batch_shape)

        self.S_past_car_frames = tf.cond(
            tf.logical_and(phi.is_training, tf.convert_to_tensor(self.rnnconf.past_perturb)), lambda: phi.S_past_car_frames_noisy, lambda: phi.S_past_car_frames)

        # To represent tensors packed to shape (flattened_batch * A_agents, ...), e.g. (B*K*A, ...)
        self.A_batch_size = self.batch_size * self.A
        self.rnn_state = self.rnn.zero_state(self.A_batch_size, dtype=tf.float64)
        self.mu_shape = self.batch_shape + (self.A, self.D)
        self.sigma_shape = self.batch_shape + (self.A, self.D, self.D)

        if len(self.batch_shape) == 2: self.batch_str = 'bk'
        elif len(self.batch_shape) == 1: self.batch_str = 'b'
        else: raise ValueError("Unhandled batch size")

        if len(self.batch_shape) == 2:
            # (B, K, A)
            yaws_batch = tensoru.expand_and_tile_axis(phi.yaws, axis=1, N=self.batch_shape[1])
            # (BKA, 1)
            self.yaws_A_batch = tf.reshape(yaws_batch, (self.A_batch_size, 1))
        elif len(self.batch_shape) == 1:
            # (BA, 1)
            self.yaws_A_batch = tf.reshape(phi.yaws, (self.A_batch_size, 1))
        else: raise ValueError("Unhandled batch size")
            
        # Create past encodings once.
        if self.past_encodings is None:
            if self.rnnconf.past_do_preconv:
                ldb("Doing preconv")
                # 1D-convolve over the past states before plugging them into the RNN.
                self.preconv_W = tf.Variable(tf.ones((self.rnnconf.preconv_horizon, self.D, self.rnnconf.past_gru_units),
                                                     dtype=tf.float64), name="W_preconv")
                self.preconv_b = tf.Variable(1e-5 * tf.ones((self.rnnconf.past_gru_units,), dtype=tf.float64), name="b_preconv")
                past_rnn_inputs = [tf.nn.conv1d(self.S_past_car_frames[:, a], self.preconv_W, stride=1, padding="SAME") + self.preconv_b
                                   for a in range(self.A)]
            else:
                ldb("Not doing preconv")
                past_rnn_inputs = [self.S_past_car_frames[:, a] for a in range(self.A)]
            # For every agent, run the RNN cell and retrieve the state at the last time step.
            self.past_encodings = [
                tf.nn.dynamic_rnn(
                    cell=self.past_rnn, inputs=past_rnn_inputs[a], initial_state=None, time_major=False, dtype=tf.float64)[1] for a in range(self.A)]
        # Create feature map once.
        if self.feature_map is None:
            if self.cnnconf.create_residual_connections:
                ldb("Creating a CNN with residual connections")
                if self.cnnconf.do_batchnorm:
                    self.feature_map = tensoru.convnet_with_residuals_and_batchnorm(
                        self.convnet, self.batchnorms, phi.overhead_features, skip_indices=set([0, self.cnnconf.n_conv_layers-1]), is_training=phi.is_training)
                else:
                    self.feature_map = tensoru.convnet_with_residuals(
                        self.convnet, phi.overhead_features, skip_indices=set([0, self.cnnconf.n_conv_layers-1]))
            else:
                ldb("Creating a CNN without residual connections")
                self.feature_map = tensoru.convnet(self.convnet, phi.overhead_features)
            if self.cnnconf.append_cnn_input_to_cnn_output:
                ldb("append_cnn_input_to_cnn_output=True")
                self.feature_map = tf.concat((self.feature_map, phi.overhead_features), axis=-1)
            if self.cnnconf.create_overhead_feature:
                # Global avg, max, min.
                self.overhead_feature = tf.concat((tf.reduce_mean(self.feature_map, axis=(-3, -2)),
                                                   tf.reduce_max(self.feature_map, axis=(-3, -2)),
                                                   tf.reduce_min(self.feature_map, axis=(-3, -2))), axis=-1)
                assert(len(batch_shape) > 1)
                tilex = functools.reduce(operator.mul, batch_shape[1:])
                self.overhead_feature = tf.tile(self.overhead_feature, (tilex, 1))
        self.feature_map_C = tensoru.size(self.feature_map, -1)
        self.radii_feat_size = self.feature_map_C * len(self.whiskerconf.radii) * self.whiskerconf.n_samples

        # Tile the past encodings for the current batch_shape
        assert(tensoru.size(phi.light_features, 0) == self.batch_shape[0])
        if len(batch_shape) > 1:
            assert(len(batch_shape) == 2)
            #[(BK, F) ... ]
            self.past_encodings_batch = [tensoru.expand_and_tile_and_pack(e, 1, 0, N=batch_shape[1]) for e in self.past_encodings]
            # (BK, F)
            light_features = tf.keras.backend.repeat_elements(phi.light_features, self.lightconf.lightrep, axis=-1)
            light_features_batch = tensoru.expand_and_tile_and_pack(light_features, 1, 0, N=batch_shape[1])
            # (BKA, F)            
            self.light_features_A_batch = tensoru.expand_and_tile_and_pack(light_features_batch, 1, 0, N=self.A)
        else:
            # [(B, F) ... ]
            self.past_encodings_batch = self.past_encodings
            # (BA, F)
            light_features_batch = tf.keras.backend.repeat_elements(phi.light_features, self.lightconf.lightrep, axis=-1)
            self.light_features_A_batch = tensoru.expand_and_tile_and_pack(light_features_batch, 1, 0, N=self.A)

        self.past_encodings_joint = []
        if self.A > 1:
            for a in range(self.A):
                others_feat = tf.reduce_sum(self.past_encodings_batch[:a] + self.past_encodings_batch[a+1:], axis=0)
                # Create a feature for each agent that depends on its own encoding and the sum of the other agent's encodings.
                self.past_encodings_joint.append(tf.concat((self.past_encodings_batch[a], others_feat), axis=-1))
            past_encodings_joint_shape = 2*self.rnnconf.past_gru_units
            self.social_map_feat_size = 2 * self.feature_map_C
        else:
            self.past_encodings_joint = [self.past_encodings_batch[0]]
            past_encodings_joint_shape = self.rnnconf.past_gru_units
            self.social_map_feat_size = self.feature_map_C

        self.past_encodings_A_batch = tf.reshape(
            tf.stack(self.past_encodings_joint, axis=-2), (self.A_batch_size, past_encodings_joint_shape))

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def check_gradients(self, z):
        # Z cannot affect the first whisker features.
        assert(tf.gradients(self.step_generate_record[0][0]['whisker_map_feats'], z)[0] is None)
        # Z must affect the later whisker features.
        assert(tf.gradients(self.step_generate_record[0][1]['whisker_map_feats'], z)[0] is not None)
        # TODO implement more checks.

    def add_crossbatch_asserts(self, outs, ins, outs_names, ins_names, n_max=100):
        assert(len(outs) == len(ins) == len(outs_names) == len(ins_names))
        for out, in_, out_name, in_name in zip(outs, ins, outs_names, ins_names):
            self.crossbatch_asserts.extend(tfutil.assert_no_crossbatch_gradients(arr_out=out, arr_in=in_, name_out=out_name, name_in=in_name, n_max=n_max))

    def step_generate(self, S_history, phi, *args, **kwargs):
        """Produce the m_t, s_t, and sig_t of the next position step.

        :param S_history: Rollout history, must be in __CAR FRAMES__.
        :param phi: ESPPhi context.
        :returns: 
        :rtype: 

        """
        ldb = lambda x: log.debug("Step {}, ".format(self.t) + x)
        S_tm1 = S_history[-1]
        # Collect some new info about this step.
        self.step_generate_record[-1][self.t] = {}

        # Last agent positions in various coordinate systems.
        S_tm1_grid = phi.local2grid.apply(S_tm1, self.batch_str + 'aj')        
        S_tm1_world = phi.local2world.apply(S_tm1, self.batch_str + 'aj')
        
        # Compute the agent features. (batch_shape, A, F)
        map_feats = bijection_helpers.get_map_feats(
            feature_map=self.feature_map, batch_shape=self.batch_shape, batch_size=self.batch_size,
            last_agent_positions_grid=S_tm1_grid, A=self.A)

        self.current_phi = phi
        self.current_local2world = phi.current_local2world
        self.current_local2grid = phi.current_local2grid

        whisker_map_feats, self.whiskers_grid = bijection_helpers.get_whisker_map_feats(
            phi=phi, feature_map=self.feature_map, batch_shape=self.batch_shape,
            batch_size=self.batch_size,
            A=self.A, 
            cars2grid=phi.current_local2grid,
            template_cars=self.whisker_template)
        
        # Save the whisker map feats for gradient checking.
        self.step_generate_record[-1][self.t]['whisker_map_feats'] = whisker_map_feats

        # len(A) list of (bsize, featsize)
        map_feats_list = tf.unstack(map_feats, axis=-2)
        social_map_feats = []
        if self.A > 1:
            for a in range(self.A):
                # Note we're summing here instead of concatenating.
                others_feat = tf.reduce_sum(map_feats_list[:a] + map_feats_list[a+1:], axis=0)

                # Create a feature for each agent that depends on its own map feat and the sum of the other agent's map feats.
                social_map_feats.append(tf.concat((map_feats_list[a], others_feat), axis=-1))
        else:
            social_map_feats.append(map_feats_list[0])

        # Compute the spatial social differences. (batch_shape, A, -1)
        last_agent_position_feats = bijection_helpers.get_social_spatial_differences(
            phi=phi, batch_shape=self.batch_shape, batch_size=self.batch_size, last_agent_positions_world=S_tm1_world, A=self.A)
        
        # Compute the social features with the MLP. (batch_size, G)
        if self.socialconf.use_social_feats:
            social_feats_all = tf.reshape(tensoru.mlp(self.social_mlp, last_agent_position_feats), (self.batch_size, self.A, self.G))
            social_feats_A_batch = tf.reshape(social_feats_all, (self.A_batch_size, self.G))

        social_map_feats_A_batch = tf.reshape(tf.stack(social_map_feats, axis=-2), (self.A_batch_size, self.social_map_feat_size))

        # Reshape to use in single RNN call.
        # Feature of last two positions of self.
        self_positions_feat_A_batch = tf.reshape(tf.stack((S_history[-2], S_history[-1]), axis=-2), (self.A_batch_size, 2*self.D))
        whisker_map_feats_A_batch = tf.reshape(whisker_map_feats, (self.A_batch_size, self.radii_feat_size))

        # Build the input to the RNN.
        joint_feat_tup = (self.past_encodings_A_batch, self.yaws_A_batch, self_positions_feat_A_batch)
        if self.lightconf.use_light_feats:
            ldb("Using light feats")
            joint_feat_tup += (self.light_features_A_batch,)
        if self.whiskerconf.use_whiskers:
            ldb("Using whiskers")
            joint_feat_tup += (whisker_map_feats_A_batch,)
        if self.socialconf.use_social_feats:
            ldb("Using social feats")
            joint_feat_tup += (social_feats_A_batch,)
        if self.socialconf.use_social_map_feats:
            ldb("Using social feats")
            joint_feat_tup += (social_map_feats_A_batch,)
        if self.cnnconf.create_overhead_feature:
            ldb("Using overhead feats")
            joint_feat_tup += (self.overhead_feature,)

        joint_feat = joint_feat_concat = tf.concat(joint_feat_tup, axis=-1)
        
        if self.mlpconf.do_prernn_mlp:
            ldb("Doing prernn mlp")
            joint_feat = tensoru.mlp(self.prernn_mlp, joint_feat)

        # Step the RNN. (batch_size, rnn_dim)
        if self.rnnconf.use_future_rnn:
            output, self.rnn_state = self.rnn(inputs=joint_feat, state=self.rnn_state)
        else:
            output = joint_feat

        if self.lightconf.use_light_feats and self.lightconf.postrnn_light_feats:
            ldb("Using postrnn light feats")
            output = tf.concat((output, self.light_features_A_batch), axis=-1)

        # Pass the RNN outputs to the MLP. (batch_size, 6)
        predictions = tensoru.mlp(self.mlp, output)


        # Build final outputs.
        m_t = tf.reshape(predictions[:, :self.D], self.mu_shape)
        sigel_t = tf.reshape(predictions[:, self.D:], self.sigma_shape)
        sigel_t = tensoru.finitize_condition_number(sigel_t)
        sigma_t = tf.linalg.expm(sigel_t)

        self_positions_feat_batch = tf.reshape(self_positions_feat_A_batch, self.mu_shape[:-1] + (-1,))
        past_encodings_batch = tf.reshape(self.past_encodings_A_batch, self.mu_shape[:-1] + (-1,))

        # For debugging.
        if self.debug_eager:
            self.step_generate_record[-1][self.t]['self_positions_feat_batch'] = self_positions_feat_batch
            self.step_generate_record[-1][self.t]['past_encodings_batch'] = past_encodings_batch        

        if self.debug_static and self.t in (0,1):
            rnn_state_batch = tf.reshape(self.rnn_state, self.mu_shape[:-1] + (-1,))
            joint_feat_batch = tf.reshape(joint_feat, self.mu_shape[:-1] + (-1,))
            joint_feat_concat_batch = tf.reshape(joint_feat_concat, self.mu_shape[:-1] + (-1,))
        
            yaws_batch = tf.reshape(self.yaws_A_batch, self.mu_shape[:-1] + (-1,))
            
            outs, outs_names, ins, ins_names = zip(*[
                # (S_tm1_grid, "S_tm1_grid_{}".format(self.t), S_tm1, "S_tm1_{}".format(self.t)),
                # (S_tm1_world, "S_tm1_world_{}".format(self.t), S_tm1, "S_tm1_{}".format(self.t)),
                # # (m_t, "m_t_{}".format(self.t), S_tm1, "S_tm1_{}".format(self.t)),
                # # (sigma_t, "sigma_t_{}".format(self.t), S_tm1, "S_tm1_{}".format(self.t)),
                # # (rnn_state_batch, "rnn_state_batch_{}".format(self.t), phi.S_past_car_frames, "S_past_car_frames"),
                # (past_encodings_batch, "past_encodings_batch_t={}".format(self.t), phi.S_past_car_frames, "S_past_car_frames"),
                # (self_positions_feat_batch, "self_posiitons_feat_t={}".format(self.t), phi.S_past_car_frames, "S_past_car_frames"),
                # (joint_feat_concat_batch, "joint_feat_concat_batch_t={}".format(self.t), phi.S_past_car_frames, "S_past_car_frames"),
                # (joint_feat_batch, "joint_feat_batch_t={}".format(self.t), phi.S_past_car_frames, "S_past_car_frames"),
                (m_t, "m_t_{}".format(self.t), phi.overhead_features, "overhead_features"),
                 # (m_t, "m_t_{}".format(self.t), phi.S_past_car_frames, "S_past_car_frames"),
                 # (m_t, "m_t_{}".format(self.t), phi.S_past_car_frames, "S_past_car_frames"),                 
            ])
            self.add_crossbatch_asserts(outs=outs,
                                        ins=ins,
                                        outs_names=outs_names,
                                        ins_names=ins_names,
                                        n_max=5)

        # Step time.
        self.t += 1
        return m_t, sigel_t, sigma_t

