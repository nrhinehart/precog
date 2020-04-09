
import os
import numpy as np
import random
import string
import tensorflow as tf

def seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

def isotropic_gaussian_entropy(dim, standard_deviation):
    """Compute the entropy of an isotropic gaussian

    :param dim: dimensionality
    :param standard_deviation: diagonal standard deviation.
    :returns: H(N(0, std**2 I_{dim}))
    :rtype: np.float64

    """
    assert(dim > 0)
    assert(standard_deviation > 0)
    return dim / 2. * np.log(2 * np.pi * np.e) + 1 / 2. * np.log((standard_deviation ** 2) ** dim)

def random_string(N_chars):
    randstate = random.getstate()
    random.seed(int.from_bytes(os.urandom(2), 'big'))
    rstring = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N_chars))
    random.setstate(randstate)
    return rstring
