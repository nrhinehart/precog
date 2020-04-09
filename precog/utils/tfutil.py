
import dill
import functools
import numpy as np
import os
import pdb
import tensorflow as tf

import precog.utils.tensor_util as tensoru

class ModelCollections:
    def __init__(self, names=['input', 'output', 'metric']):
        self.names = names

        # Maintain a dictionary that maps the names to collections (lists of tensors)
        def _update_collections():
            self.tensor_collections = {name: tf.compat.v1.get_collection(name) for name in self.names}
        
        for name in names:
            setattr(self, name, set())

            def _(x, name):
                # Prevent adding non-Tensors to a collection
                assert(isinstance(x, tf.Tensor))
                xset = getattr(self, name)
                if x not in xset:
                    xset.add(x)
                    tf.compat.v1.add_to_collection(name, x)
                    # Update the tensor collections.
                    _update_collections()
            setattr(self, 'add_{}'.format(name), functools.partial(_, name=name))

class FeedDict(dict):
    def __setitem__(self, k, v):
        ks = tensoru.shape(k)
        vs = tensoru.shape(v)
        if ks != vs:
            raise ValueError("Cannot assign to key of shape {} with value of shape {}".format(ks, vs))
        return dict.__setitem__(self, k, v)

    def validate(self):
        for k, v in self.items():
            ks = tensoru.shape(k)
            vs = tensoru.shape(v)
            if ks != vs:
                raise ValueError("Feed will fail because of key of shape {} has value of shape {}".format(ks, vs))
    
def create_session(allow_growth=True, per_process_gpu_memory_fraction=.2):
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False,
                                                                allow_soft_placement=False,
                                                                device_count={},
                                                                gpu_options=gpu_options))
    return sess

def load_model(model_path, sess, checkpoint_path):
    assert(os.path.isdir(model_path))
    if checkpoint_path in ('None', None):
        checkpoint_path = tf.compat.v1.train.latest_checkpoint(model_path)
    saver = tf.compat.v1.train.import_meta_graph(checkpoint_path + '.meta')
    saver.restore(sess, checkpoint_path)
    return checkpoint_path, sess.graph

def load_annotated_model(model_path, sess, checkpoint_path=None):
    with open(model_path + '/collections.dill', 'rb') as f:
        collection_names = dill.load(f)

    ckpt, graph = load_model(model_path, sess, checkpoint_path=checkpoint_path)
    tensor_collections = {name: tf.compat.v1.get_collection(name) for name in collection_names}
    return ckpt, graph, tensor_collections

def get_collection_dict(tensor_list):
    return {_.name.split(':')[0]: _ for _ in tensor_list}

def get_multicollection_dict(dict_of_collections):
    joint_coll_dict = {}
    for k, coll_list in dict_of_collections.items():
        joint_coll_dict.update(get_collection_dict(coll_list))
    return joint_coll_dict

def get_tensor_names(graph=None):
    graph = graph or tf.get_default_graph()
    return [t.name for op in graph.get_operations() for t in op.values()]

def get_tensors(graph=None):
    graph = graph or tf.get_default_graph()
    return [t for op in graph.get_operations() for t in op.values()]

def interpolate_bilinear(grid, query_points, indexing="ij", name=None):
    """
    COPIED from https://github.com/tensorflow/addons/blob/320ad67895b99fd244949b1faba868fce05b3b9f/tensorflow_addons/image/dense_image_warp.py

    Unfortunately, tensorflow_addons is not compatible with tf versions < 2.0. So I copied the code instead.

    Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.
    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape
        `[batch, N, 2]`.
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).
      name: a name for the operation (optional).
    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`
    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the
        inputs invalid.
    """
    if indexing != "ij" and indexing != "xy":
        raise ValueError("Indexing mode must be \'ij\' or \'xy\'")

    with tf.name_scope(name or "interpolate_bilinear"):
        grid = tf.convert_to_tensor(grid)
        query_points = tf.convert_to_tensor(query_points)

        if len(grid.shape) != 4:
            msg = "Grid must be 4 dimensional. Received size: "
            raise ValueError(msg + str(grid.shape))

        if len(query_points.shape) != 3:
            raise ValueError("Query points must be 3 dimensional.")

        if query_points.shape[2] is not None and query_points.shape[2] != 2:
            raise ValueError("Query points must be size 2 in dim 2.")

        if grid.shape[1] is not None and grid.shape[1] < 2:
            raise ValueError("Grid height must be at least 2.")

        if grid.shape[2] is not None and grid.shape[2] < 2:
            raise ValueError("Grid width must be at least 2.")

        grid_shape = tf.shape(grid)
        query_shape = tf.shape(query_points)

        batch_size, height, width, channels = (grid_shape[0], grid_shape[1],
                                               grid_shape[2], grid_shape[3])

        shape = [batch_size, height, width, channels]

        # pylint: disable=bad-continuation
        with tf.control_dependencies([
                tf.compat.v1.debugging.assert_equal(
                    query_shape[2],
                    2,
                    message="Query points must be size 2 in dim 2.")
        ]):
            num_queries = query_shape[1]
        # pylint: enable=bad-continuation

        query_type = query_points.dtype
        grid_type = grid.dtype

        # pylint: disable=bad-continuation
        with tf.control_dependencies([
                tf.compat.v1.debugging.assert_greater_equal(
                    height, 2, message="Grid height must be at least 2."),
                tf.compat.v1.debugging.assert_greater_equal(
                    width, 2, message="Grid width must be at least 2."),
        ]):
            alphas = []
            floors = []
            ceils = []
            index_order = [0, 1] if indexing == "ij" else [1, 0]
            unstacked_query_points = tf.unstack(query_points, axis=2)
        # pylint: enable=bad-continuation

        for dim in index_order:
            with tf.name_scope("dim-" + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = shape[dim + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = tf.constant(0.0, dtype=query_type)
                floor = tf.math.minimum(
                    tf.math.maximum(min_floor, tf.math.floor(queries)),
                    max_floor)
                int_floor = tf.cast(floor, tf.dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = tf.cast(queries - floor, grid_type)
                min_alpha = tf.constant(0.0, dtype=grid_type)
                max_alpha = tf.constant(1.0, dtype=grid_type)
                alpha = tf.math.minimum(
                    tf.math.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = tf.expand_dims(alpha, 2)
                alphas.append(alpha)

        # pylint: disable=bad-continuation
        with tf.control_dependencies([
                tf.compat.v1.debugging.assert_less_equal(
                    tf.cast(
                        batch_size * height * width, dtype=tf.dtypes.float32),
                    np.iinfo(np.int32).max / 8.0,
                    message="The image size or batch size is sufficiently "
                    "large that the linearized addresses used by tf.gather "
                    "may exceed the int32 limit.")
        ]):
            flattened_grid = tf.reshape(
                grid, [batch_size * height * width, channels])
            batch_offsets = tf.reshape(
                tf.range(batch_size) * height * width, [batch_size, 1])
        # pylint: enable=bad-continuation

        # This wraps tf.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using tf.gather_nd.
        def gather(y_coords, x_coords, name):
            with tf.name_scope("gather-" + name):
                linear_coordinates = (
                    batch_offsets + y_coords * width + x_coords)
                gathered_values = tf.gather(flattened_grid, linear_coordinates)
                return tf.reshape(gathered_values,
                                  [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], "top_left")
        top_right = gather(floors[0], ceils[1], "top_right")
        bottom_left = gather(ceils[0], floors[1], "bottom_left")
        bottom_right = gather(ceils[0], ceils[1], "bottom_right")

        # now, do the actual interpolation
        with tf.name_scope("interpolate"):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (
                bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp

def safe_det(x, *args, **kwargs):
    """Computes determinants of a batch of matrices. Allows for some matrices to be singular.

    :param x: 
    :returns: 
    :rtype: 

    """
    raise NotImplementedError("For some reason, scatter_nd fails sometimes even when the input shapes are correct")
    xs = tensoru.shape(x)
    assert(xs[-1] == xs[-2])
    # The shape of the matrices
    D = xs[-1]
    # Flatten batch dim.
    x_flat = tf.reshape(x, (-1, D, D))
    # Use SVD to count number of nonzero singular values -> invertible matrices have D nonsingular values.
    nonsingular_matrix_bools = tf.equal(tf.reduce_sum(tf.cast(tf.linalg.svd(x_flat, compute_uv=False) > 1.5e-7, tf.int32), axis=-1), D)
    # Get the indices of the invertible matrices
    nonsingular_matrix_inds = tf.cast(tf.squeeze(tf.where(nonsingular_matrix_bools)), tf.int32)
    # Gather the invertible matrices
    x_flat_nonsingular = tf.gather(x_flat, nonsingular_matrix_inds)
    # Compute determinants of the invertible matrices
    det_x_nonsingular_matrix = tf.linalg.det(x_flat_nonsingular)
    # Scatter the nonzero determinants into a vector. Unreference entries default to 0.
    dets = tf.scatter_nd(indices=nonsingular_matrix_inds, updates=det_x_nonsingular_matrix, shape=tf.shape(nonsingular_matrix_bools))
    # Reshape the resulting determinants.
    dets_full = tf.reshape(dets, shape=tensoru.shape(x)[:-2])
    return dets_full

def require_complete_parameterization(params):
    """Require params ^ TRAINABLE_VARIABLES = TRAINABLE_VARIABLES

    :param params: list of tf.Variable
    """
    all_variables = set([_ for _ in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if _.name.find("batch_norm") == -1])
    params = set(params)
    if all_variables != params:
        unnecessary = params - all_variables
        missing = all_variables - params
        raise ValueError("Params do not form a complete parameterization!\nMissing: {}\nUnnecessary: {}".format(missing, unnecessary))

def compute_crossbatch_gradient(arr_out, arr_in, b_in, b_out, axis=0):
    # TODO tmp
    assert(b_in != b_out)
    if axis == 0:
        out_vars = arr_out[b_out]
        partial_out_partial_in = tf.gradients(out_vars, arr_in)
        if partial_out_partial_in is None or partial_out_partial_in[0] is None:
            raise ValueError("There are no gradients whatsoever!")
        else:
            assert(isinstance(partial_out_partial_in, list))
            assert(len(partial_out_partial_in) == 1)
            return partial_out_partial_in[0][b_in]
    elif axis == 1:
        out_vars = arr_out[:, b_out]
        partial_out_partial_in = tf.gradients(out_vars, arr_in)
        if partial_out_partial_in is None or partial_out_partial_in[0] is None:
            raise ValueError("There are no gradients whatsoever!")
        else:
            assert(isinstance(partial_out_partial_in, list))
            assert(len(partial_out_partial_in) == 1)
            return partial_out_partial_in[0][:, b_in]
    else:
        raise ValueError("unhandled axis")
    
    return partial_out_partial_in

def assert_no_crossbatch_gradients(arr_out, arr_in, axis=0, n_max=100, name_out='', name_in=''):
    shape = tensoru.shape(arr_out)
    shape1 = tensoru.shape(arr_in)
    bdim = shape[axis]
    bdim1 = shape1[axis]
    # Ensure batch dimensionality is the same.
    assert(bdim == bdim1)
    # TODO temp.
    assert(bdim == 10)
    checks = []
    for b_in in range(bdim):
        for b_out in range(bdim):
            if b_in == b_out: continue
            grad = compute_crossbatch_gradient(arr_out=arr_out, arr_in=arr_in, b_in=b_in, b_out=b_out, axis=axis)
            checks.append(tf.compat.v1.assert_near(grad, 0.0, message="Cross-batch gradient (d[{}]_b={}/d[{}]_b={})is nonzero".format(
                name_out, b_in, name_in, b_out)))
            if len(checks) > n_max: break
    return checks
