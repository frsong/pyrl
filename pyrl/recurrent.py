from   collections import OrderedDict
import sys

import numpy as np

import theano
from   theano import tensor

class Recurrent(object):
    """
    Generic recurrent unit.

    """
    def __init__(self, type_, name=''):
        self.type   = type_
        self.name   = self.type
        self.params = OrderedDict()
        self.masks  = {}

        if name != '':
            self.name += '-' + name

        # self.N
        # self.trainables
        # self.f_hidden
        # self.f_out
        # self.f_log_out
        # self.step

    @property
    def noise_dim(self):
        return self.N

    def get_dim(self, name):
        raise NotImplementedError

    def index(self, name):
        for i, trainable in enumerate(self.trainables):
            if trainable.name == name:
                return i
        return None

    def get_masks(self):
        return {k: v.get_value() for k, v in self.masks.items()}

    def get_values(self):
        return OrderedDict([(k, v.get_value()) for k, v in self.params.items()])

    def get(self, name):
        p = self.params[name]
        if name in self.masks:
            #return tensor.nnet.relu(p)
            return self.masks[name]*p
        return p

    def func_step_0(self, use_x0=False):
        """
        Returns a Theano function.

        """
        if use_x0:
            x0 = tensor.vector('x0')
        else:
            x0 = self.get('x0')
        Wout = self.get('Wout')
        bout = self.get('bout')

        r = self.f_hidden(x0)
        z = self.f_out(r.dot(Wout) + bout)

        args = []
        if use_x0:
            args += [x0]

        return theano.function(args, [z, x0])

    def func_step_t(self):
        """
        Returns a Theano function.

        """
        Wout = self.get('Wout')
        bout = self.get('bout')

        inputs = tensor.matrix('inputs')
        noise  = tensor.matrix('noise')
        x_tm1  = tensor.matrix('x_tm1')

        x_t = self.step(inputs, noise, x_tm1, *self.step_params)
        r_t = self.f_hidden(x_t)
        z_t = self.f_out(r_t.dot(Wout) + bout)

        return theano.function([inputs, noise, x_tm1], [z_t[0], x_t[0]])

    def get_outputs_0(self, x0, log=False):
        Wout = self.get('Wout')
        bout = self.get('bout')
        r0   = self.f_hidden(x0)

        if log:
            return self.f_log_out(r0.dot(Wout) + bout)
        return self.f_out(r0.dot(Wout) + bout)

    def get_outputs(self, inputs, noise, x0, log=False):
        Wout = self.get('Wout')
        bout = self.get('bout')

        x, _ = theano.scan(self.step,
                           outputs_info=[x0],
                           sequences=[inputs, noise],
                           non_sequences=self.step_params)
        r = self.f_hidden(x)

        if log:
            return x, self.f_log_out(r.dot(Wout) + bout)
        return x, self.f_out(r.dot(Wout) + bout)

    def get_regs(self, x0_, x, M):
        return 0
