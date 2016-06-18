import numpy as np

import theano
from   theano import tensor

import theanotools

class Adam(object):
    def __init__(self, trainables, accumulators=None):
        self.trainables = trainables

        if accumulators is None:
            self.means = [theanotools.shared(0*x.get_value()) for x in trainables]
            self.vars  = [theanotools.shared(0*x.get_value()) for x in trainables]
            self.time  = theanotools.shared(0)
        else:
            self.means = [theanotools.shared(x) for x in accumulators[0]]
            self.vars  = [theanotools.shared(x) for x in accumulators[1]]
            self.time  = theanotools.shared(accumulators[2])

    def get_values(self):
        means  = [x.get_value() for x in self.means]
        vars_  = [x.get_value() for x in self.vars]
        time   = self.time.get_value()

        return [means, vars_, time]

    def get_updates(self, loss, lr, max_norm=1, beta1=0.9, beta2=0.999,
                    epsilon=1e-8, grads=None):
        # Gradients
        if grads is None:
            grads = tensor.grad(loss, self.trainables)

        # Clipping
        norm  = tensor.sqrt(sum([tensor.sqr(g).sum() for g in grads]))
        m     = theanotools.clipping_multiplier(norm, max_norm)
        grads = [m*g for g in grads]

        # Safeguard against numerical instability
        new_cond = tensor.or_(tensor.or_(tensor.isnan(norm), tensor.isinf(norm)),
                              tensor.or_(norm < 0, norm > 1e10))
        grads = [tensor.switch(new_cond, np.float32(0), g) for g in grads]

        # Safeguard against numerical instability
        #cond  = tensor.or_(norm < 0, tensor.or_(tensor.isnan(norm), tensor.isinf(norm)))
        #grads = [tensor.switch(cond, np.float32(0), g) for g in grads]

        # New values
        t       = self.time + 1
        lr_t    = lr*tensor.sqrt(1. - beta2**t)/(1. - beta1**t)
        means_t = [beta1*m + (1. - beta1)*g for g, m in zip(grads, self.means)]
        vars_t  = [beta2*v + (1. - beta2)*tensor.sqr(g) for g, v in zip(grads, self.vars)]
        steps   = [lr_t*m_t/(tensor.sqrt(v_t) + epsilon)
                   for m_t, v_t in zip(means_t, vars_t)]

        # Updates
        updates  = [(x, x - step) for x, step in zip(self.trainables, steps)]
        updates += [(m, m_t) for m, m_t in zip(self.means, means_t)]
        updates += [(v, v_t) for v, v_t in zip(self.vars, vars_t)]
        updates += [(self.time, t)]

        return norm, grads, updates
