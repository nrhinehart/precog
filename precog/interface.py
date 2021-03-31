
import attrdict
import copy
import collections
import functools
import logging
import numpy as np
import operator
import pdb
import six
import tensorflow as tf
import types

from abc import ABCMeta, abstractproperty, abstractmethod

import precog.utils.class_util as classu
import precog.utils.log_util as logu
import precog.utils.similarity_util as similarityu
import precog.utils.tensor_util as tensoru
import precog.utils.tfutil as tfutil

log = logging.getLogger(__file__)

@six.add_metaclass(ABCMeta)
class ESPOptimizer:
    @abstractmethod
    def optimize(self, distribution, dataset, *args, **kwargs):
        """ Runs the optimization and evaluation procedures."""        
        pass

@six.add_metaclass(ABCMeta)    
class ESPDataset:
    @abstractmethod
    def get_minibatch(self, mb_idx=None, *args, **kwargs): pass

    @abstractmethod
    def get_T(self): pass

    @abstractproperty
    def name(self): pass

    @abstractproperty
    def max_A(self): pass

@six.add_metaclass(ABCMeta)    
class ESPDistribution:
    @abstractmethod
    def sample(self, phi, *args, **kwargs): pass

    @abstractmethod
    def log_prob(self): pass

    @abstractproperty
    def variables(self): pass

    @abstractproperty
    def A(self): pass

@six.add_metaclass(ABCMeta)    
class ESPJointTrajectoryBijection:
    @abstractmethod
    def forward(self): pass

    @abstractmethod
    def inverse(self): pass

    @abstractproperty
    def variables(self): pass

    @abstractproperty
    def current_metadata(self): pass

    @abstractmethod
    def _prepare(self, batch_shape, phi): pass

    def crossbatch_asserts(): return []

@six.add_metaclass(ABCMeta)
class ESPObjective:
    @classu.member_initialize
    def __init__(self, K_perturb=12, perturb=True, perturb_epsilon=1e-2): pass
    
    @abstractmethod
    def call(self, model_distribution, model_distribution_samples, target_distribution_minibatch, data_epsilon, target_distribution_proxy):
        pass

    @abstractproperty
    def requires_samples(self): pass

    def __call__(self, *args, **kwargs): return self.call(*args, **kwargs)    

@six.add_metaclass(ABCMeta)        
class ProxyDistribution:
    @abstractmethod
    def log_prob(self, samples, minibatch, *args, **kawrgs): pass

@six.add_metaclass(ABCMeta)
class ESPSampleMetric:
    @abstractmethod
    def call(self, model_distribution_samples, data_distribution_samples): pass

    @abstractmethod
    def __repr__(self): pass

    def __call__(self, *args, **kwargs): return self.call(*args, **kwargs)
    
class ESPObjectiveReturn:
    @classu.member_initialize
    def __init__(self, min_criterion, forward_cross_entropy, ehat, rollout, reverse_cross_entropy=None):
        if self.reverse_cross_entropy is None: self.reverse_cross_entropy = tf.convert_to_tensor(-np.inf)

    def unpack(self):
        return [self.min_criterion, self.forward_cross_entropy, self.ehat, self.rollout, self.reverse_cross_entropy]

@six.add_metaclass(ABCMeta)    
class Numpyable:
    @abstractmethod
    def to_numpy(self, sessrun): raise RuntimeError
    
@six.add_metaclass(ABCMeta)    
class NumpyableTensorGroup(Numpyable):
    @abstractproperty
    def tensor_names(self):
        """User implements a list of strings corresponding to the gettable tf.Tensor member attributes of a class."""

    @property
    def tensors(self): return [getattr(self, _) for _ in self.tensor_names]

    @logu.log_wrapi(False)
    def to_numpy(self, sessrun):
        """Mixin method to convert a class represention in tensors to numpy, using a bound sess.run(-, feed_dict) function.

        :param sessrun: output of funtools.partial(sess.run, feed_dict=feed_dict)
        :returns: 
        :rtype: 

        """
        assert(len(self.tensors) == len(self.tensor_names))        
        rets_and_names = zip(sessrun(self.tensors), self.tensor_names)
        np_self = copy.copy(self)
        for r, n in rets_and_names: setattr(np_self, n, r)
        return np_self

@six.add_metaclass(ABCMeta)
class NumpyableTensorGroupGroup(Numpyable):
    @abstractproperty
    def tensor_group_names(self): pass

    @classu.classproperty
    def tensor_groups(self):
        return [getattr(self, _) for _ in self.tensor_group_names]

    @logu.log_wrapi(False)
    def to_numpy(self, sessrun):
        """Mixin method to convert a class represention in tensors to numpy, using a bound sess.run(-, feed_dict) function.

        :param sessrun: output of funtools.partial(sess.run, feed_dict=feed_dict)
        :returns: 
        :rtype: 

        """
        group_names = self.tensor_group_names
        tensor_to_group_name = {} 
        np_groups = {}
        names = []
        tensors = []
        
        for group_name in group_names:
            group = getattr(self, group_name)
            np_groups[group_name] = copy.copy(group)
            tensors.extend(group.tensors)
            names.extend(group.tensor_names)
            for t in group.tensors:
                assert(t not in tensor_to_group_name)
                tensor_to_group_name[t] = group_name
            
        assert(len(names) == len(tensors) == len(tensor_to_group_name))
        
        rets_and_names_and_objs = zip(sessrun(tensors), names, tensors)
        np_self = copy.copy(self)
        for tensor_np, tensor_name, tensor in rets_and_names_and_objs:
            # Get the name of this tensor's group
            group_name = tensor_to_group_name[tensor]
            np_group = np_groups[group_name]
            # Update the group the numpy ret.
            setattr(np_group, tensor_name, tensor_np)
            # Update the groupgroup with the updated group
            setattr(np_self, group_name, np_group)
        return np_self
                       
class ESPBaseSamplesAndLogQ(NumpyableTensorGroup):
    @classu.member_initialize
    def __init__(self, Z_sample, log_q_samples):
        pass
        # assert(Z_sample.name.find("Z_sample") >= 0)
        # assert(log_q_samples.name.find("log_q_samples") >= 0)

    @classu.classproperty
    def tensor_names(self): return ['Z_sample', "log_q_samples"]

class ESPRollout(NumpyableTensorGroup):
    @classu.member_initialize
    def __init__(self, S_car_frames_list, Z_list, m_list, mu_list, sigma_list, sigel_list, metadata_list, phi):
        """
        NB that S_car_frames_list is in the original coordinate frame at t=0. 
        """
        # (..., A, T, D) [i.e. (..., Agents, Timesteps, State Dimension) ]

        # Stack along time.
        self.S_car_frames = tf.stack(S_car_frames_list, axis=-2, name='S_car_frames')
        self.Z = tf.stack(Z_list, axis=-2, name='Z')
        self.m = tf.stack(m_list, axis=-2, name='m')
        self.mu = tf.stack(mu_list, axis=-2, name='mu')
        self.sigma = tf.stack(sigma_list, axis=-3, name='sigma')
        self.sigel = tf.stack(sigel_list, axis=-3, name='sigel')

        # Convert coords.
        self.S_world_frame = tf.identity(phi.local2world.apply(self.S_car_frames), name='S_world_frame')
        self.S_grid_frame = tf.identity(phi.world2grid.apply(self.S_world_frame), name='S_grid_frame')

        # Metadata.
        self.shape = tensoru.shape(self.Z)
        self.event_shape = self.shape[-3:]
        self.batch_shape = self.shape[:-3]
        
        # The dimensionality of the rollout.
        self.dimensionality = self.event_size = functools.reduce(operator.mul, self.event_shape)
        self.rollout_outputs = set(self.tensors)

        # Represent car frames -> grid coordinates.
        #self.rollout_car_frames_list_grid = [self.phi.world2grid * _ for _ in rollout_car_frames_list]

    @classu.classproperty
    def tensor_names(self):
        return ['S_car_frames', 'S_world_frame', 'S_grid_frame', 'Z', 'm', 'mu', 'sigma', 'sigel']
        # For now let's just use these.
        # return ['S_car_frames', 'S_world_frame', 'S_grid_frame']
        
    def __repr__(self):
        return self.__class__.__name__ + "(batch_shape={}, event_shape={})".format(self.batch_shape, self.event_shape)

class ESPPhi(NumpyableTensorGroup):
    @classu.tensor_member_initialize
    def tensor_init(self, S_past_world_frame, yaws, overhead_features, agent_presence, light_strings, is_training): pass

    @classu.member_initialize
    def __init__(self,
                 S_past_world_frame,
                 yaws,
                 overhead_features,
                 agent_presence,
                 feature_pixels_per_meter,
                #  is_training,
                 is_training=False,
                 light_strings=None,
                 yaws_in_degrees=True,
                 past_perturb_epsilon=5e-2,
                 name=False):
        """

        :param S_past_world_frame: (B, A, T, D)
        :param yaws: (B, A)
        :param overhead_features: (B, H, W, C)
        :param agent_presence: (B, A)
        :param yaws_in_degrees: bool
        """
        assert(feature_pixels_per_meter >= 1)
        if light_strings is None:
            light_strings = np.array(['NONE']*tensoru.size(yaws, 0), dtype=np.unicode_)
            
        # Overwrite these members with tensorized versions of them.        
        self.tensor_init(S_past_world_frame, yaws, overhead_features, agent_presence, light_strings, is_training)
        
        self._frames_init()

        past_noise_world_frame = tf.random.normal(
            mean=0.,
            stddev=self.past_perturb_epsilon,
            shape=tensoru.shape(self.S_past_world_frame),
            dtype=tf.float64)
        # Always create an alternative noisy-past (learning may not use it).
        self.S_past_world_frame_noisy = self.S_past_world_frame + past_noise_world_frame

        # (B, A, T, D)
        self.S_past_car_frames = self.world2local.apply(self.S_past_world_frame)
        self.S_past_car_frames_noisy = self.world2local.apply(self.S_past_world_frame_noisy)
        # (B, A, T, D)
        self.S_past_grid_frame = self.world2grid.apply(self.S_past_world_frame, dtype=tf.float64)
        self.S_past_grid_frame_noisy = self.world2grid.apply(self.S_past_world_frame_noisy, dtype=tf.float64)
        
        self.light_features = one_hotify_light_strings(self.light_strings)

        if name:
            # Name some intermediate computations (not placeholders)
            self.S_past_grid_frame = tf.identity(self.S_past_grid_frame, name='S_past_grid_frame')
            self.S_past_car_frames = tf.identity(self.S_past_car_frames, name='S_past_car_frames')
            self.light_features = tf.identity(self.light_features, name='light_features')

    def _frames_init(self):
        self.local2world, self.world2local, self.world2grid, self.local2grid = ESPPhi.frames_init(
            self.S_past_world_frame,
            self.overhead_features,
            self.yaws,
            self.yaws_in_degrees,
            self.feature_pixels_per_meter)

    @staticmethod
    def frames_init(S_past_world_frame, overhead_features, yaws, yaws_in_degrees, feature_pixels_per_meter):
        agent_origins_world_frame = S_past_world_frame[..., -1, :]

        # Object to convert from local vehicle frames to world frame.        
        local2world = similarityu.SimilarityTransform.from_origin_and_rotation(
            origin=agent_origins_world_frame, theta=yaws, degrees=yaws_in_degrees, scale=1.)
        
        # Object to convert from world frame to frames of each vehicle.
        world2local = local2world.invert()

        H, W = tensoru.shape(overhead_features)[-3:-1]
        world_origin_grid_frame = tf.constant((H//2, W//2), dtype=tf.float64)
        # Object to convert from world frame to grid frame
        world2grid = similarityu.SimilarityTransform.from_origin_and_rotation(
            world_origin_grid_frame, tf.constant(0.,dtype=tf.float64), scale=feature_pixels_per_meter)
        # N.B. this transformation means:
        #  1. The world origin is in the center of the overhead features
        #  2. There's no rotation ->
        #    2a. the traj_x coordinate indexes into the _WIDTH_ dimension of the feature map.
        #    2b. the traj_y coordinate indexes into the _HEIGHT_ dimension of the feature map

        # Object to convert from local frames of each vehicle to grid frame.
        local2grid = world2grid * local2world
        return (local2world, world2local, world2grid, local2grid)

    def prepare(self, batch_shape):
        if len(batch_shape) == 2: self.batch_str = 'bk'
        elif len(batch_shape) == 1: self.batch_str = 'b'
        else: raise ValueError("Unhandled batch size")
        
        if len(batch_shape) == 1:            
            self.current_local2world = self.original_local2world = self.world2local.invert()
            # Input spaces: Car frames. Output space: grid frame.
            self.current_local2grid = self.original_local2grid = self.world2grid * self.current_local2world
        else:
            R = tensoru.repeat_expand_dims(self.world2local.R, axis=1, n=len(batch_shape) - 1)
            R = tf.tile(R, (1,) + batch_shape[1:] + (1,) * (tensoru.rank(self.world2local.R) - 1))
            t = tensoru.repeat_expand_dims(self.world2local.t, axis=1, n=len(batch_shape) - 1)
            t = tf.tile(t, (1,) + batch_shape[1:] + (1,) * (tensoru.rank(self.world2local.t) - 1))
            # Use this object to track the car frames during the rollout.
            # Input space: world coordinates. Output spaces: car frame coordinates.
            world2local = similarityu.SimilarityTransform(R=R, t=t, scale=self.world2local.scale)
            # "current_local2<X>" frames track the frame as it rolls out. The "local" frames are fixed, however.
            self.current_local2world = self.original_local2world = world2local.invert()
            # Input spaces: Car frames. Output space: grid frame.
            self.current_local2grid = self.original_local2grid = self.world2grid * self.current_local2world

    def update_frames(self, S_t_car_frames, S_tm1_car_frames):
        """TODO verify and visualize that this makes sense when A > 1 (i.e. when local_t0 != world)!!!

        :param S_t_car_frames: In car frames
        :returns: 
        :rtype: 

        """
        S_t_world_frame = self.original_local2world.apply(S_t_car_frames, self.batch_str + 'aj')
        velocity_in_car_frame = S_t_car_frames - S_tm1_car_frames

        
        velocity_in_world_frame = self.original_local2world.apply(velocity_in_car_frame, self.batch_str + 'aj', translate=False)
        vx = velocity_in_world_frame[..., 0]
        vy = velocity_in_world_frame[..., 1]
        heading_in_world_frame = tf.atan(vy/vx)

        local_frame_R_vel = similarityu.SimilarityTransform._Rtheta(heading_in_world_frame, lib=tf)
        local_frame_R_prev = self.current_local2world.R

        # Hack to try to prevent spinning when agent is nearly stationary.
        nontrivial_motion_mask = tf.linalg.norm(velocity_in_car_frame,axis=-1) > 0.1
        nontrivial_motion_mask_tile = tf.tile(nontrivial_motion_mask[..., None, None], (1,) * (len(self.batch_str) + 1) + (2, 2))
        local_frame_R = tf.where(nontrivial_motion_mask_tile, local_frame_R_vel, local_frame_R_prev)
        self.current_local2world = similarityu.SimilarityTransform(R=local_frame_R, t=S_t_world_frame, scale=self.current_local2world.scale)
        self.current_local2grid = self.world2grid * self.current_local2world
        
    def __repr__(self):
        return self.__class__.__name__ + "(...)"

    @classu.classproperty
    def tensor_names(self):
        return (['S_past_world_frame', 'yaws', 'overhead_features', 'agent_presence', 'S_past_car_frames', 'S_past_grid_frame',
                 "light_strings", "is_training"])

def one_hotify_light_strings(light_strings):
    """Featurize light strings in the tensorflow graph. By putting it in the graph, we don't have to remember how they're featurized
    when it comes to inference time.

    :param light_strings: 
    :returns: 
    :rtype: 

    """
    light_strings = tf.strings.upper(light_strings)
    assert(tensoru.rank(light_strings) == 1)
    eye = tf.eye(5, dtype=tf.float64)[..., None]
    c = lambda x: tf.cast(x, tf.float64)
    none = c(tf.equal(light_strings, 'NONE'))[None]
    green = c(tf.equal(light_strings, 'GREEN'))[None]
    yellow = c(tf.equal(light_strings, 'YELLOW'))[None]
    intersection = c(tf.equal(light_strings, 'INTERSECTION'))[None]
    red = c(tf.equal(light_strings, 'RED'))[None]

    # Ensure there's exactly one catgoirzation for each.
    with tf.control_dependencies([tf.compat.v1.assert_equal(none + green + yellow + intersection + red, tf.ones_like(none))]):
        # This puts them in separate bins.
        feats = tf.transpose(eye[0] * none + eye[1] * green + eye[2] * intersection + eye[3] * yellow + eye[4] * red)
        # This puts them all in the same bin.        
        # feats = tf.transpose(eye[0] * none + eye[0] * green + eye[0] * intersection + eye[0] * yellow + eye[4] * red)
    return feats
    
class MetadataItem:
    @classu.member_initialize
    def __init__(self, name, array, dtype):
        pass

class MetadataList(list):
    @property
    def arrays(self): return [_.array for _ in self]

    @property
    def names(self): return [_.name for _ in self]

    def to_dict(self): return collections.OrderedDict(zip(self.names, self.arrays))
    
class ESPPhiMetadata(NumpyableTensorGroup):
    @classu.member_initialize
    def __init__(self, phi, metadata_list, name=False):
        name = None if not name else 'agent_counts'
        assert(isinstance(self.metadata_list, MetadataList))
        assert(all([isinstance(_, MetadataItem) for _ in metadata_list]))
        self.agent_counts = tf.reduce_sum(tf.cast(phi.agent_presence, tf.float64), axis=1, name=name)
        B0 = self.phi.S_past_car_frames.shape[0]
        B1 = self.phi.yaws.shape[0]
        B2 = self.phi.overhead_features.shape[0]
        B3 = self.phi.agent_presence.shape[0]
        assert(B0 == B1 == B2 == B3)
        self.B = B0.value
        self.H = tensoru.size(self.phi.overhead_features, 1)
        if len(self.metadata_list):
            self.tensor_init(**self.metadata_list.to_dict())

    @classu.tensor_member_initialize
    def tensor_init(self, **kwargs): pass
        
    def __repr__(self):
        return self.__class__.__name__ + "(B={}, agent_counts=...)".format(self.B)

    @classu.classproperty
    def tensor_names(self):
        # TODO missing some...
        return ['agent_counts'] # + self.metadata_list.names

class ESPExperts(NumpyableTensorGroup):
    @classu.member_initialize
    def __init__(self, S_future_car_frames, S_future_world_frame, S_future_grid_frame): pass

    def __repr__(self): return self.__class__.__name__ + "(...)"

    @classu.classproperty
    def tensor_names(self):
        return ['S_future_car_frames', 'S_future_world_frame', 'S_future_grid_frame']

class ESPExpertsMetadata(NumpyableTensorGroup):
    @classu.member_initialize
    def __init__(self, training):
        self.B = self.training.S_future_car_frames.shape[0]
        self.T = self.training.S_future_car_frames.shape[-2]
        assert(self.B == self.training.S_future_world_frame.shape[0])
        
    def __repr__(self): return self.__class__.__name__ + "(B={})".format(self.B)

    @classu.classproperty
    def tensor_names(self): return []

class ESPTrainingInput(NumpyableTensorGroupGroup):
    placeholders = []
    sample_placeholders = []
    classcount = 0
    
    @classu.member_initialize
    def __init__(self, phi, phi_m, experts, experts_m):
        """Holds either ndarrays or tf.Tensors

        :param phi: 
        :param phi_m: 
        :param experts: 
        :param experts_m: 
        :returns: 
        :rtype: 

        """
        assert(phi_m.B == experts_m.B)
    
    def __repr__(self):
        return self.__class__.__name__ + "(phi={}, phi_m={}, experts={}, experts_m={}".format(
            self.phi, self.phi_m, self.experts, self.experts_m)

    @classmethod
    def from_data(cls, S_past_world_frame, yaws, overhead_features, agent_presence, S_future_world_frame, feature_pixels_per_meter,
                  light_strings,
                  is_training,
                  metadata_list=None,
                  name=False):

        # These objects convert most of their inputs to tensors.
        phi = ESPPhi(
            S_past_world_frame,
            yaws,
            overhead_features,
            agent_presence,
            feature_pixels_per_meter,
            name=name,
            light_strings=light_strings,
            is_training=is_training)
        
        phi_m = ESPPhiMetadata(phi, name=name, metadata_list=metadata_list)
        S_future_world_frame = tf.convert_to_tensor(S_future_world_frame)
        S_future_car_frames = phi.world2local.apply(S_future_world_frame)
        S_future_grid_frame = phi.world2grid.apply(S_future_world_frame, dtype=tf.float64)

        if name:
            S_future_car_frames = tf.identity(S_future_car_frames, name='S_future_car_frames')
            S_future_grid_frame = tf.identity(S_future_grid_frame, name='S_future_grid_frame')
        experts = ESPExperts(
            S_future_car_frames=S_future_car_frames, S_future_world_frame=S_future_world_frame, S_future_grid_frame=S_future_grid_frame)
        experts_m = ESPExpertsMetadata(experts)
        cls.classcount += 1
        return cls(phi, phi_m, experts, experts_m)

    def to_singleton(self):
        """Creates a new object with the tensor inputs replaced by placeholders."""
        
        assert(self.classcount == 1)

        #******* Placeholder creation. *******
        S_past_world_frame = tf.compat.v1.placeholder(
            shape=self.phi.S_past_world_frame.shape, dtype=self.phi.S_past_world_frame.dtype, name='S_past_world_frame')
        
        yaws = tf.compat.v1.placeholder(shape=self.phi.yaws.shape, dtype=self.phi.yaws.dtype, name='yaws')
        
        overhead_features = tf.compat.v1.placeholder(
            shape=self.phi.overhead_features.shape, dtype=self.phi.overhead_features.dtype, name='overhead_features')
        
        agent_presence = tf.compat.v1.placeholder(
            shape=self.phi.agent_presence.shape, dtype=self.phi.agent_presence.dtype, name='agent_presence')
        
        S_future_world_frame = tf.compat.v1.placeholder(
            shape=self.experts.S_future_world_frame.shape, dtype=self.experts.S_future_world_frame.dtype, name='S_future_world_frame')

        light_strings = tf.compat.v1.placeholder(
            shape=self.phi.light_strings.shape, dtype=self.phi.light_strings.dtype, name="light_strings")

        is_training = tf.compat.v1.placeholder(shape=self.phi.is_training.shape, dtype=self.phi.is_training.dtype, name="is_training")

        metadata_placeholders = [tf.compat.v1.placeholder(shape=_.array.shape, dtype=_.dtype, name=_.name) for _ in self.phi_m.metadata_list]
        ESPTrainingInput.placeholders = [S_past_world_frame,
                                         yaws,
                                         overhead_features,
                                         agent_presence,
                                         light_strings,
                                         S_future_world_frame,
                                         is_training] + metadata_placeholders
        expert_index = ESPTrainingInput.placeholders.index(S_future_world_frame)
        
        # We don't need (or want) the future when sampling.
        ESPTrainingInput.sample_placeholders = ESPTrainingInput.placeholders[:expert_index] + ESPTrainingInput.placeholders[expert_index+1:]
        ESPTrainingInput.classcount += 1
        return ESPTrainingInput.from_data(
            S_past_world_frame=S_past_world_frame,
            yaws=yaws,
            overhead_features=overhead_features,
            agent_presence=agent_presence,
            S_future_world_frame=S_future_world_frame,
            feature_pixels_per_meter=self.phi.feature_pixels_per_meter,
            light_strings=light_strings,
            is_training=is_training,
            name=True,
            metadata_list=self.phi_m.metadata_list)

    def to_feed_dict(self, S_past_world_frame, yaws, overhead_features, agent_presence, S_future_world_frame, metadata_list, light_strings, is_training):
        fd = tfutil.FeedDict(
            zip(self.placeholders,
                [S_past_world_frame, yaws, overhead_features, agent_presence, light_strings, S_future_world_frame, is_training] + metadata_list.arrays))
        fd.validate()
        return fd
        
    @classu.classproperty
    def tensor_group_names(self): return ['phi', 'phi_m', 'experts', 'experts_m']
    
class ESPTestInput(NumpyableTensorGroupGroup):
    @classu.member_initialize
    def __init__(self, phi, phi_m):
        """This class only used at inference time (not during the training program)

        :param phi: 
        :param phi_m: 
        :returns: 
        :rtype: 

        """
        
    @classmethod
    def from_data(cls, pasts, yaws, overhead_features, agent_presence, feature_pixels_per_meter, is_training):
        phi = ESPPhi(pasts, yaws, overhead_features, agent_presence, feature_pixels_per_meter, is_training)
        phi_m = ESPPhiMetadata(phi)
        return cls(phi, phi_m)

class ESPSampledOutput(NumpyableTensorGroupGroup):
    @classu.member_initialize
    def __init__(self, rollout, phi, phi_metadata, base_and_log_q): pass

    @classu.classproperty
    def tensor_group_names(self): return ['rollout', 'phi', 'phi_metadata', 'base_and_log_q']

    def __repr__(self): return self.__class__.__name__ + "(rollout={})".format(self.rollout)

class Mock:
    def __init__(self, name, kv):
        """A class used to mock another class

        :param name: 
        :param kv: dict: member (str) -> value (Object)
        :returns: 
        :rtype: 

        """
        self.name = name
        for k, v in kv.items():
            setattr(self, k, v)

    def __repr__(self): return "{}Mock()".format(self.name)    

class MockNumpyableTensorGroup(NumpyableTensorGroup):
    def __init__(self, name, kv):
        """A class used to mock another class

        :param name: 
        :param kv: dict: member (str) -> value (Object)
        :returns: 
        :rtype: 

        """
        self.name = name
        self._tensor_names = []
        for k, v in kv.items():
            setattr(self, k, v)
            # Assumes kv is a dict of str:tf.Tensor
            self._tensor_names.append(k)

    @property
    def tensor_names(self):
        return self._tensor_names

    def __repr__(self): return "{}Mock()".format(self.name)

class ESPInference:
    def __init__(self, tensor_collections):
        """Use the tensors from the graph to mock objects that were used to build the graph.

        :param tensor_collections: dict: collection_name (str) -> collection (list)
        :returns: 
        :rtype: 

        """
        
        self.joint_coll_dict = tfutil.get_multicollection_dict(tensor_collections)
        
        # Mock the objects we used when we originally constructed the graph.
        self.add('phi', ESPPhi)
        # Sampled rollout
        self.add('rollout', ESPRollout)
        # Sampled Z's
        self.add('base_and_log_q', ESPBaseSamplesAndLogQ)
        self.add('phi_metadata', ESPPhiMetadata)
        self.add('experts', ESPExperts)
        self.add('experts_metadata', ESPExpertsMetadata)

        # This object is important to construct -- we'll need it for plotting.
        self.sampled_output = ESPSampledOutput(
            rollout=self.rollout, phi=self.phi, phi_metadata=self.phi_metadata, base_and_log_q=self.base_and_log_q)
        self.training_input = Mock(
            ESPTrainingInput.__name__, {'phi': self.phi, 'phi_m': self.phi_metadata, 'experts': self.experts, 'experts_m': self.experts_metadata})
        self.test_input = Mock(
            ESPTestInput.__name__, {'phi': self.phi, 'phi_m': self.phi_metadata})

        # Create a to_feed_dict member function, modelled after the real `to_feed_dict` function.
        # TODO this is annoying and brittle to hardcode to match the to_feed_dict above. 
        def to_feed_dict(xself, S_past_world_frame, yaws, overhead_features, agent_presence, light_strings, S_future_world_frame, metadata_list, is_training):
            return dict(zip(xself.placeholders,
                            [S_past_world_frame, yaws, overhead_features, agent_presence, light_strings, S_future_world_frame, is_training]))
        # Uses the fact that this collection was created in order...
        self.training_input.placeholders = tensor_collections['infer_input']
        self.training_input.to_feed_dict = types.MethodType(to_feed_dict, self.training_input)
        self.sampled_output.placeholders = tensor_collections['sample_input']

        # TODO some params are hardcoded.
        self.phi.local2world, self.phi.world2local, self.phi.world2grid, self.phi.local2grid = ESPPhi.frames_init(
            self.phi.S_past_world_frame, self.phi.overhead_features, self.phi.yaws, yaws_in_degrees=True, feature_pixels_per_meter=2)
        
        # Hack in missing things here.
        self.phi_metadata.B, self.phi_metadata.H, *_ = tensoru.shape(self.phi.overhead_features)
        # Store some sample-rollout shape metadata
        B, K, A, T, D = tensoru.shape(self.sampled_output.rollout.S_world_frame)
        self.metadata = attrdict.AttrDict({'B': B, 'K': K, 'A': A, 'T': T, 'D': D})

    def add(self, name, klass):
        """Creates a mock object for the provided class.

        :param name: name of the target attribute
        :param klass: class instance
        :param members_attr: attribute that can be used to retrieve the desired member names of the target class
        """
        
        tensors_dict = {}
        names = getattr(klass, 'tensor_names')
        # Retrieve the tensor from the graph that the given klass expects.
        for n in names:
            t = self.joint_coll_dict.get(n, None)
            if t is None: log.warning("ESPInference's member object is missing a member! '{}'. Class={}".format(n, klass))
            tensors_dict[n] = t
        # Instatiate a mocked version of the klass using the tensors we retrieved from the graph.
        mocked = MockNumpyableTensorGroup(klass.__name__, tensors_dict)
        setattr(self, name, mocked)
