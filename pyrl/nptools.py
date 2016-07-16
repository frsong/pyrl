import numpy as np

def get_rng(seed=0, loc='Not specified'):
    print("<< RNG {} >> {}".format(seed, loc))

    return np.random.RandomState(seed)

def relu(x):
    return np.maximum(0, x)
