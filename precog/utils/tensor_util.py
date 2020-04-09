
import collections
import functools
import numpy as np
import operator
import tensorflow as tf

def np_tf_get_shape(inp):
    """Get the shape of the array

    :param inp: ndarray or tf.Tensor
    :returns: shape of input
    :rtype: tuple 
    """
    if isinstance(inp, np.ndarray):
        return inp.shape
    else:
        return tf_shape(inp)

def np_tf_get_a0(inp): return np_tf_get_shape(inp)[0]

def np_tf_get_ndim(inp): return len(np_tf_get_shape(inp))

def np_tf_dimension_assert(inp, dims):
    """Array dimensionality check

    :param inp:
    :param dim:
    :returns:
    :rtype:

    """
    try:
        assert(inp.ndim in dims)
    except AttributeError:
        assert(len(tf_shape(inp)) in dims)
    except TypeError:
        assert(inp.ndim == dims)

def np_tf_shape_assert(inp, shape):
    """Array shape check

    :param inp:
    :param shape:
    :returns:
    :rtype:

    """
    try:
        assert(inp.shape == shape)
    except AttributeError:
        assert(tuple(tf_shape(inp)) == shape)

def np_tf_axis_assert(inp, axis, target):
    """Array axis length check

    :param inp:
    :param axis:
    :param target:
    :returns:
    :rtype:

    """

    try:
        assert(inp.shape[axis] == target)
    except AttributeError:
        assert(tf_shape(inp)[axis] == target)
    return True

def np_tf_transpose(inp):
    try:
        return inp.T
    except AttributeError:
        return tf.transpose(inp)

def size(inp, dim=None, axis=None):
    """Return the size of a specific axis of a tf.Tensor as a Python int. If dim is not provided or is None,

    :param inp: tf.Tensor
    :param dim: int axis index
    :param axis: alias of axis
    :returns: size of Tensor along axis
    :rtype: int

    """
    if dim is None and axis is None: return functools.reduce(operator.mul, shape(inp))
    elif axis is None: return np_tf_get_shape(inp)[dim]
    else: return np_tf_get_shape(inp)[axis]

def tf_shape(inp):
    """Get the shape of a Tensor

    :param inp: tf.Tensor
    :returns: shape of tensor
    :rtype: tuple
    """

    return tuple(inp.get_shape().as_list())

def shape(inp): return np_tf_get_shape(inp)

def rank(inp):
    """Returns a python int of the rank / number of dimensions of a tf.Tensor.
    Note tf.rank is similar, except it returns a tensor, which we need to
    evaluate if we want to use its value...

    :param inp: tf.Tensor or np.ndarray
    :returns: r \in \mathbb N_0
    :rtype: int

    """
    return len(shape(inp))

def get_swapping_permutation(r, idx0, idx1):
    """Build a permutation that swaps two indices

    :param r: rank of the permutation
    :param idx0: source index
    :param idx1: target index
    :returns: sequence of permutation indices
    :rtype: list

    """
    assert(0 <= idx0)
    assert(0 <= idx1)
    permute = list(range(r))
    permute[idx0] = idx1
    permute[idx1] = idx0
    return permute

def pidx(r, idx):
    """Get a nonnegative index for the given rank and index

    :param r: maximum rank int \in N_+
    :param idx: idx \in [-r, r-1]
    :returns: idx \in [0, r-1]
    """
    
    if r < 1: raise ValueError("R is not positive")
    if not (-r <= idx <= r - 1): raise IndexError("Index is OOB")
    if idx < 0: return r + idx
    else: return idx

def get_popinsert_permutation(r, idx0, idx1):
    """Build a permutation that cycles a subcycle to put idx0 at idx

    :param r: rank of array
    :param idx0: index of axis to pop
    :param idx1: index before which to put the popped axis
    :returns: sequence of permutation indices
    :rtype: list

    """
    idx0 = pidx(r, idx0)
    idx1 = pidx(r, idx1)
    permute = list(range(r))
    permute.insert(idx1, permute.pop(idx0))
    return permute

def swap_axes(arr, axis0, axis1, lib=tf, **kwargs):
    """Transpose an array to swap axes. Doesn't require full knowledge of the dimensionality of array.

    :param arr: array object (e.g. tf.Tensor)
    :param axis0: int in [-rank, rank - 1]
    :param axis1: int in [-rank, rank - 1]
    :param lib: library to call transpose with (e.g. tf or np)
    :returns: array object
    """
    r = rank(arr)
    axis0 = pidx(r, axis0)
    axis1 = pidx(r, axis1)
    assert(axis0 <= r - 1), "rank oob"
    assert(axis1 <= r - 1), "rank oob"
    return lib.transpose(arr, get_swapping_permutation(r, axis0, axis1), **kwargs)

def popinsert_axes(arr, axis0, axis1, lib=tf, **kwargs):
    """popinsert_axes((A, B, C, D, E), 1, 4) -> (A, C, D, E, B)

    :param arr: 
    :param axis0: int in [-rank, rank - 1]
    :param axis1: int in [-rank, rank - 1]
    :returns: 
    :rtype: 

    """
    r = rank(arr)
    axis0 = pidx(r, axis0)
    axis1 = pidx(r, axis1)
    assert(-r <= axis0 <= r - 1), "rank oob"
    assert(-r <= axis1 <= r - 1), "rank oob"
    return lib.transpose(arr, get_popinsert_permutation(r, axis0, axis1), **kwargs)

def pack_to_axis(arr, axis_source, axis_target, lib=tf, **kwargs):
    """Pack the source axis with the target axis with the source serving as the outer
    axis (rows) and target serving as inner (cols)

    :param arr: 
    :param axis_source: 
    :param axis_target: 
    :returns: 
    :rtype: 

    """

    r = rank(arr)
    s = np_tf_get_shape(arr)
    axis_source = pidx(r, axis_source)
    axis_target = pidx(r, axis_target)
    assert(-r <= axis_source <= r - 1), "axis source oob"
    assert(axis_target != axis_source), "identical axes"
    assert(-r + 1 <= axis_target <= r - 1), "axis target oob"
    fwd = axis_source < axis_target
    pretarget = axis_target - fwd
    cycled = popinsert_axes(arr, axis_source, pretarget, lib=lib)
    permuter = get_popinsert_permutation(r, axis_source, pretarget)
    if None in s:
        ts_arr = tf.shape(arr)
        s_permute = [ts_arr[_] for _ in permuter]
        pack_permute = tf.concat((s_permute[:pretarget], [s_permute[pretarget] * s_permute[pretarget+1]], s_permute[pretarget+2:]), axis=0)
        return lib.reshape(cycled, pack_permute, **kwargs)
    else:
        s_permute = np.asarray(s)[permuter].tolist()
        del s_permute[pretarget:pretarget + 2]
        s_permute.insert(pretarget, -1)
        return lib.reshape(cycled, s_permute, **kwargs)

def unpack_axis(arr, axis_source, axis1_size, lib=tf, **kwargs):
    """unpack_axis((A, B, CD, E), 2, D) -> (A, B, C, D, E) 

    :param arr: 
    :param axis_source: 
    :param axis1_size: the size of the second axis (D in the example)
    :returns: 
    :rtype: 

    """
    r = rank(arr)
    axis_source = pidx(r, axis_source)
    s = list(np_tf_get_shape(arr))
    assert(0 <= axis_source <= r - 1), "axis source oob"
    del s[axis_source]
    s_unpack = s[:axis_source] + [-1, axis1_size] + s[axis_source:]
    return lib.reshape(arr, s_unpack, **kwargs)

def frontpack(arr, lib=tf, **kwargs):
    """ (A, B, ...) -> (AB, ...) """
    return pack_to_axis(arr, 0, 1, lib=lib, **kwargs)

def frontswap(arr, lib=tf, **kwargs):
    """ (A, B, ...) -> (B, A, ...)"""
    return swap_axes(arr, 0, 1, lib=lib, **kwargs)

def backpack(arr, lib=tf, **kwargs):
    """ (..., A, B) -> (..., AB)"""
    return pack_to_axis(arr, -2, -1, lib=lib, **kwargs)

def backswap(arr, lib=tf, **kwargs):
    """ (..., A, B) -> (..., B, A)"""
    return swap_axes(arr, -2, -1, lib=lib, **kwargs)

def rotate_left(arr, lib=tf, **kwargs):
    """ (A, B, ..., Y, Z) -> (B, ..., Y, Z, A)"""
    return popinsert_axes(arr, 0, -1, lib=lib, **kwargs)

def rotate_right(arr, lib=tf, **kwargs):
    """ (A, B, ..., Y, Z) -> (Z, A, B, ..., Y)"""
    return popinsert_axes(arr, -1, 0, lib=lib, **kwargs)

def expand_and_tile_and_pack(arr, expand_ax, pack_ax, N=1, lib=tf, **kwargs):
    arr_exp = expand_and_tile_axis(arr, expand_ax=0, N=N, lib=lib, **kwargs)
    return pack_to_axis(arr_exp, expand_ax, pack_ax, lib=lib)

def expand_and_tile_axis(arr, axis=0, N=1, lib=tf, **kwargs):
    # TODO check if it works with lib=np
    r = rank(arr)
    arr_expanded = lib.expand_dims(arr, axis=axis)
    tiler = [1] * (r + 1)
    tiler[axis] = N
    return lib.tile(arr_expanded, tiler)

def matrices_to_back(arr, check_square=False):
    """

    :param arr: (M, N, ...)
    :returns: (..., M, N)
    :rtype: 

    """
    if check_square: assert(size(arr, 0) == size(arr, 1))
    return popinsert_axes(popinsert_axes(arr, 0, -1), 0, -1)

def matrices_to_front(arr, check_square=False):
    """FIXME! briefly describe function

    :param arr: (..., M, N)
    :returns: (M, N, ...)
    :rtype: 

    """
    if check_square: assert(size(arr, -2) == size(arr, -1))
    return popinsert_axes(popinsert_axes(arr, -1, 0), -1, 0)

def repeat_expand_dims(arr, n, axis=0):
    return functools.reduce(lambda arr_, ax: tf.expand_dims(arr_, axis=ax), [axis]*n, arr)

def repeat_reduce_sum(arr, n, axis=-1):
    return functools.reduce(lambda arr_, ax: tf.reduce_sum(arr_, axis=ax), [axis]*n, arr)

def tile_to_batch(arr, batch_shape):
    expanded = repeat_expand_dims(arr, len(batch_shape), axis=0)
    return tf.tile(expanded, batch_shape + (1,)*rank(arr))

def mlp(layers, arr):
    res = arr
    for layer in layers: res = layer(res)
    return res

def convnet(layers, arr):
    res = arr
    for layer in layers: res = layer(res)
    return res

def convnet_with_residuals(layers, arr, skip_indices=set()):
    res = arr
    for i, j in zip(range(len(layers) - 1), range(1, len(layers))):
        layer_i, layer_j = layers[i], layers[j]
        layer_j_out = layer_j(layer_i(res))
        if i in skip_indices: res = layer_j_out
        else: res = res + layer_j_out
    return res

def convnet_with_residuals_and_batchnorm(layers, bn_layers, arr, is_training, skip_indices=set()):
    res = arr
    assert(len(layers) == len(bn_layers))
    for i, j in zip(range(len(layers) - 1), range(1, len(layers))):
        # Add residual every 2 layers.
        layer_i, layer_j = layers[i], layers[j]
        bn_i, bn_j = bn_layers[i], bn_layers[j]
        layer_j_out = bn_j(layer_j(bn_i(layer_i(res), training=is_training)), training=is_training)
        if i in skip_indices: res = layer_j_out
        else: res = res + layer_j_out
    return res

def layer_variables(layers):
    var = []
    for layer in layers: var += layer.variables
    return var

def finitize_condition_number(matrix, eps=1e-8, dtype=tf.float64):
    *batch_shape, N, M = shape(matrix)[-2:]
    return matrix + eps * tf.eye(num_rows=N, num_columns=M, batch_shape=batch_shape, dtype=dtype)

def isclose(arr0, arr1, atol=1e-8, rtol=1e-8):
    return tf.less_equal(tf.abs(arr0 - arr1), atol + rtol * tf.abs(arr1))
