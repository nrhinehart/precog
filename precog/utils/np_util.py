

import functools
import inspect
import numpy as np
import pdb
import scipy.ndimage.morphology as morph
import sys
import tensorflow as tf

import precog.utils.tensor_util as tensoru


this = sys.modules[__name__]

def entropy_lower_bound(k, stddev):
    """Get entropy lower bound of a pdf perturbed with isotropic gaussian noise at the provided stddev.

    :param k: Dimensionality of isotropic gaussian
    :param stddev: Standard deviation of the distribution

    """
    return k / 2. + k / 2. * np.log(2 * np.pi) + 1 / 2. * k * np.log(stddev ** 2)

# Create numpy-specialized versions of all of the functions that have a 'lib' argument.
for o in inspect.getmembers(tensoru):
    if inspect.isfunction(o[1]):
        args = inspect.getargspec(o[1])
        if 'lib' in args.args:
            func_np = functools.partial(o[1], lib=np)            
            setattr(this, o[0], func_np)

def batch_center_crop(arr, target_h, target_w):
    *_, h, w, _ = arr.shape
    center = (h // 2, w // 2)
    return arr[..., center[0] - target_h // 2:center[0] + target_h // 2, center[1] - target_w // 2:center[1] + target_w // 2, :]

def signed_distance_transform(binary_image, normalize=True, clip_top=1, clip_bottom=-10, dtype=np.float64):
    """

    :param binary_image: np.ndarray with boolean type and two dimensions.
    :param normalize: whether to normalize the result to [0, 1]
    :param clip_top: if normalizing, positive distance at which to clip
    :param clip_bottom: if normalizing, negative distance at which to clip
    :param dtype: dtype of the 
    :returns: 
    :rtype: 

    """
    assert(binary_image.dtype is np.dtype(np.bool))
    assert(tensoru.rank(binary_image) == 2)
    
    dt = morph.distance_transform_edt
    sdt = (dt(binary_image) - dt(1-binary_image)).astype(dtype)
    if not normalize:
        return sdt
    else:
        assert(clip_top > clip_bottom)
        assert(clip_top >= 1)
        sdt[sdt > clip_top] = clip_top
        sdt[sdt < clip_bottom] = clip_bottom
        return (sdt - clip_bottom) / (clip_top - clip_bottom)

def fill_axis_to_size(arr, axis, size, fill=0., clip=False):
    if arr.shape[axis] > size:
        if clip:
            return np.take(arr, axis=axis, indices=range(0, size))
        else:
            raise ValueError("Axis too large!")
    elif arr.shape[axis] == size:
        return arr
    else:
        diff = size - arr.shape[axis]
        new_shape = list(arr.shape)
        new_shape[axis] = diff
        return np.concatenate((arr, fill * np.ones(new_shape, dtype=np.float32)), axis=axis)

def lock_nd(arr):
    """
    Marks the ndarray as read-only.
    """
    arr.flags.writeable = False
    return arr

def unlock_nd(arr):
    """
    Marks the ndarray as writeable.
    """
    arr.flags.writeable = True
    return arr

def tabformat(arr, n=1):
    pref = '\n' + '\t'*n
    return pref + '{}'.format(arr).replace('\n', pref)
