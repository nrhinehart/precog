
import functools
import numpy as np
import pdb
import tensorflow as tf

import precog.utils.class_util as classu
import precog.utils.tensor_util as tensoru

class SimilarityTransform:
    @classu.member_initialize
    def __init__(self, R, t, scale, lib=tf):
        """
        Implements x' = f(x; R, t, scale) = scale * R * x + t for batched data.

        t represents the origin of the transformed coordinates.
        R represents the pre-translation rotation matrix.

        :param R: (B, A, 2, 2) or (B, 2, 2) or (2, 2)
        :param t: (B, A, 2) or (B, 2) or (2,)
        :param scale: scalar
        :returns: 
        :rtype: 
        
        """
        assert(lib in (np, tf))
        self.rr = tensoru.rank(R)
        self.tr = tensoru.rank(t)
        # self.theta = lib.atan2(R[..., 1, 0], R[..., 0, 0])
        joint_rank = (self.rr, self.tr)
        assert(joint_rank in [(5,4), (4,3), (3,2), (2,1)])
        if self.rr == 5:   self._ein = 'bkaij'        
        elif self.rr == 4: self._ein = 'baij'
        elif self.rr == 3: self._ein = 'bij'
        elif self.rr == 2: self._ein = 'ij'
        else: raise ValueError("Unhandled Rotation matrix dimensionality!")

        # Maintain some tiled versions for broadcasted adding.
        self._ts = {}
        if self.tr == 3:
            self._ts['baj'] = self.t
            self._ts['batj'] = tensoru.repeat_expand_dims(self.t, 1, axis=-2)
            self._ts['bkaj'] = tensoru.swap_axes(self._ts['batj'], 1, 2)
            self._ts['bkatj'] = tensoru.repeat_expand_dims(self._ts['batj'], 1, axis=1)
            self._ts['baNj'] = self.t[..., None, :]
        elif self.tr == 4:
            self._ts['bkaj'] = self.t
            self._ts['batj'] = tensoru.swap_axes(self._ts['bkaj'], 1, 2)
            self._ts['bkaNj'] = self.t[..., None, :]
            self._ts['bkaNj'] = self.t[..., None, :]
        else:
            pass
        
    def apply(self, points, points_ein=None, dtype=None, name=None, translate=True):
        input_rank = tensoru.rank(points)
        if dtype is None: dtype = self.lib.float64
        assert(dtype in (self.lib.float64, self.lib.float32, self.lib.int32))

        if points_ein is None:
            if input_rank == 5: points_ein = 'bkatj'
            # Note that we can't handle 'bkaj' input implicitly... i.e. force the second batching to be the second axis.
            elif input_rank == 4: points_ein = 'batj'
            elif input_rank == 3: points_ein = 'baj'
            elif input_rank == 2: points_ein = 'bj'
            elif input_rank == 1: points_ein = 'j'
            else: raise ValueError("Unhandled points dimensionality: {}".format(input_rank))

        # Ensure input only uses certain characters
        assert(len(set(points_ein) - set('bkatjN')) == 0)

        einstr = '{},{}->{}'.format(self._ein, points_ein, points_ein.replace('j','i'))
        points_R = self.scale * self.lib.einsum(einstr, self.R, points)

        # Add the translation.
        if translate:
            if self.tr == 1: points_Rt = points_R + self.t
            else: points_Rt = points_R + self._ts[points_ein]
        else:
            points_Rt = points_R

        if dtype == self.lib.int32: return self.lib.cast(self.lib.round(points_Rt), self.lib.int32)
        else: return points_Rt

    @classmethod
    def _Rtheta(cls, theta, lib):
        Rs = lib.stack([(lib.cos(theta), -lib.sin(theta)), (lib.sin(theta), lib.cos(theta))], axis=0)
        # (x, y, ...) -> (..., x, y)                    
        if tensoru.rank(Rs) > 2: Rs = tensoru.rotate_left(tensoru.rotate_left(Rs, lib=lib), lib=lib)
        return Rs
    
    @classmethod
    def from_origin_and_rotation(cls, origin, theta, degrees=True, scale=1., lib=None):
        """Builds a coordinate frame transformation that transforms points in its frame to points in the canonicial frame

        i.e. `origin` is a point in canonical frame, `theta` is the direction of the x axis in frame B, and 
        this method returns a transformation from points in frame B to the canonical frame

        :param cls: 
        :param origin: 
        :param theta: 
        :param degrees: 
        :param scale: 
        :returns: 
        :rtype: 

        """
        if lib is None: lib = tf
        if degrees: theta *= np.pi / 180.
        Rs = cls._Rtheta(theta, lib)
        return cls(R=Rs, t=origin, scale=scale, lib=lib)

    def invert(self):
        """
        x' = scale * R * x + t
        -> 
        x = 1/scale * R^T x' - 1/scale * R^T * b
        x = scale' * R' * x' + b' = f(x' ; R', t', scale') = f(x'; R^T, -1/scale * R^T b, 1/scale)

        :returns: 
        :rtype: 

        """
        R_T = tensoru.backswap(self.R)

        return SimilarityTransform(R_T, - 1. / self.scale * self.lib.einsum('...ij,...j->...i', R_T, self.t), scale=1./self.scale)

    def __mul__(self, other):
        """ self(other(x)) = g(y) = g(sRx + t) = s'R'(sRx + t) + t' = s'*s R'*R x + (s'R't + t') """
        s_new = self.scale * other.scale
        R_new = self.lib.einsum('...ij,...jk->...ik', self.R, other.R)
        t_new = self.scale * self.lib.einsum('...ij,...j->...i', self.R, other.t) + self.t
        return SimilarityTransform(R=R_new, t=t_new, scale=s_new)

#    def update_from_motion_input(self, new_origin_input_frame, jitter_thresh=1e-2):
        """
        Requires new origin in the input frame of the coordinate system.
        # .apply() requires BKAD
        if tensoru.rank(new_origin_input_frame) == 4: axswap = lambda a: tensoru.swap_axes(a, -3, -2)
        else: axswap = lambda a: a
        new_origin_input_frame_BKAD = axswap(new_origin_input_frame)
        new_origin_output_frame = axswap(self.invert().apply(new_origin_input_frame_BKAD))
        return self.update_from_motion_output(new_origin_output_frame, jitter_thresh=jitter_thresh)
        """

    def update_from_motion_input(self, new_origin_input_frame, jitter_thresh=1e-2):
        """
        Requires new origin in the output frame of the coordinate transformation.
        """
        #        translated = SimilarityTransform(R=self.R, t=new_origin_output_frame, scale=self.scale)
        new_origin_output_frame = new_origin_input_frame - self.t
        delta_t_local = new_origin_output_frame
        
        vx = v_heading = delta_t_local[..., 0]
        vy = v_sideways = delta_t_local[..., 1]
        zd = tf.cast(0.0, tf.float64)
        
        is_jittered = self.lib.logical_or(
            tensoru.isclose(v_sideways, zd, atol=jitter_thresh), tensoru.isclose(v_heading, zd, atol=jitter_thresh))
        # If theres nonzero x movement and y movement in the local coordinate frame, the frame has rotated.
        # If only in <1 dimension, then there's been no rotation.
        #  N.B. this could do weird things if the motion is above `jitter_thresh` and vy/vx is large (large sideways motion)
        angle = self.lib.where(is_jittered, tf.zeros_like(vx), self.lib.atan(vy/vx))
        # Rotation matrix for the angle.
        delta_R_local = self._Rtheta(angle, lib=self.lib)
        # Rotate to align with the new local x orientation along the velocity vector.
        R_new = self.lib.einsum('...ij,...jk->...ik', delta_R_local, self.R)
        # .t requires BKAD
        # We've rotated the frame and translated it to a new point in the output space.
        #return SimilarityTransform(R=R_new, t=new_origin_output_frame, scale=self.scale)
        return SimilarityTransform(R=self.R, t=new_origin_output_frame, scale=self.scale)

    def to_numpy(self):
        """NB: Only works in eager mode currently.

        :returns: 
        :rtype: 

        """
        if self.lib == np: return self
        else: return SimilarityTransform(R=self.R.numpy(), t=self.t.numpy(), scale=self.scale, lib=np)
