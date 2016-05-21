from __future__ import absolute_import, division

import sys
from   collections import OrderedDict

import numpy as np

import theano
from   theano import tensor

from .          import matrixtools, nptools, theanotools
from .recurrent import Recurrent

configs_required = ['Nin', 'Nout']
configs_default  = {
    'N':       50,
    'rho':     1.5,
    'f_out':   'softmax',
    'L2_r':    0.002,
    'L1_Wrec': 0,
    'L2_Wrec': 0,
    'fix':     []
    }

class Simple(Recurrent):
    """
    Simple units.

    """
    def __init__(self, config, params=None, seed=1):
        super(Simple, self).__init__('gru')

        #---------------------------------------------------------------------------------
        # Config
        #---------------------------------------------------------------------------------

        self.config = {}

        # Required
        for k in configs_required:
            if k not in config:
                print("[ Simple ] Error: {} is required.".format(k))
                sys.exit()
            self.config[k] = config[k]

        # Defaults available
        for k in configs_default:
            if k in config:
                self.config[k] = config[k]
            else:
                self.config[k] = configs_default[k]

        #---------------------------------------------------------------------------------
        # Activations
        #---------------------------------------------------------------------------------

        # Hidden
        self.f_hidden        = tensor.nnet.relu
        self.states_to_rates = nptools.relu

        # Output
        if self.config['f_out'] == 'softmax':
            self.f_out  = theanotools.softmax
            self.f_out3 = theanotools.softmax3
        elif self.config['f_out'] == 'linear':
            self.f_out  = (lambda x: x)
            self.f_out3 = self.f_out
        else:
            raise NotImplementedError("[ Simple ] Unknown output activation {}."
                                      .format(self.config['f_out']))

        #---------------------------------------------------------------------------------
        # Initialize parameters
        #---------------------------------------------------------------------------------

        Nin  = self.config['Nin']
        N    = self.config['N']
        Nout = self.config['Nout']

        if params is None:
            # Random number generator
            print("Seed = {}".format(seed))
            rng = np.random.RandomState(seed)

            # Network parameters
            params = OrderedDict()
            params['Win']      = rng.normal(size=(Nin, N))
            params['bin']      = np.zeros(N)
            params['Wrec']     = rng.normal(size=(N, N))
            params['Wout']     = config.get('Wout_init', np.zeros((N, Nout)))
            params['bout']     = np.zeros(Nout)
            params['states_0'] = config.get('states_0_init', np.arctanh(0.5))*np.ones(N)

            # Scale for weight initialization
            rho = self.config['rho']

            # Spectral radius
            W3 = params['Wrec']
            for W in [W3]:
                W *= rho/matrixtools.spectral_radius(W)

        # Share
        self.params = OrderedDict()
        for k in params:
            self.params[k] = theanotools.shared(params[k], k)

        # Trainable parameters
        self.trainables = [self.params[k]
                           for k in self.params if k not in self.config['fix']]

        #---------------------------------------------------------------------------------
        # Setup
        #---------------------------------------------------------------------------------

        # Dimensions
        self.Nin  = Nin
        self.N    = N
        self.Nout = Nout

        # E/I mask
        if False:
            self.Mrec_gates  = np.ones_like(params['Wrec_gates'])
            self.Mrec_states = np.ones_like(params['Wrec_states'])

        # Leak
        self.alpha = config['dt']/config['dt']

        # Regularizers
        self.L1_Wrec = self.config['L1_Wrec']
        self.L2_Wrec = self.config['L2_Wrec']
        self.L2_r    = self.config['L2_r']

        print("alpha = {}".format(self.alpha))
        print("L2_r = {}".format(self.L2_r))

        #---------------------------------------------------------------------------------
        # Define a step
        #---------------------------------------------------------------------------------

        def step(inputs, noise, states, alpha, Win, bin, Wrec):
            inputs_t     = inputs.dot(Win) + bin
            state_inputs = inputs_t

            r = self.f_hidden(states)

            next_states = r.dot(Wrec) + state_inputs + noise
            next_states = (1 - alpha)*states + alpha*next_states

            return next_states

        self.step         = step
        self.step_params  = [self.alpha]
        self.step_params += [self.params[k]
                             for k in ['Win', 'bin', 'Wrec']]

    def get_regs(self, states_0_, states, M):
        """
        Additional regularization terms.

        """
        regs = 0

        if self.L1_Wrec > 0:
            W = self.params['Wrec']
            regs += self.L1_Wrec * tensor.mean(abs(W))

        if self.L2_Wrec > 0:
            W = self.params['Wrec']
            regs += self.L2_Wrec * tensor.mean(tensor.sqr(W))

        #---------------------------------------------------------------------------------
        # Firing rates
        #---------------------------------------------------------------------------------

        if self.L2_r > 0:
            baseline = 0.

            M_ = (tensor.tile(M.T, (states.shape[-1], 1, 1))).T
            states_all = tensor.concatenate(
                [states_0_.reshape((1, states_0_.shape[0], states_0_.shape[1])), states],
                axis=0
                )
            r = self.f_hidden(states_all)
            regs += self.L2_r * tensor.sum(tensor.sqr(r - baseline)*M_)/tensor.sum(M_)

        #---------------------------------------------------------------------------------

        return regs

    def get_dOmega_dWrec(self, loss, x):
        # Pascanu's trick
        scan_node = x.owner.inputs[0].owner
        assert isinstance(scan_node.op, theano.scan_module.scan_op.Scan)
        npos   = scan_node.op.n_seqs + 1
        init_x = scan_node.inputs[npos]
        g_x    = theano.grad(loss, init_x)

        # To force immediate derivatives
        d_xt = T.tensor3('d_xt')
        xt   = T.tensor3('xt')

        # Vanishing-gradient regularization
        self.bound        = 1e-20
        self.lambda_Omega = 2

        # Wrec
        Wrec = self.params['Wrec']

        # Numerator
        alpha = self.alpha
        num   = (1 - alpha)*d_xt[1:] + T.dot(alpha*d_xt[1:], Wrec.T)*self.df_hidden(xt)
        num   = (num**2).sum(axis=2)

        # Denominator
        denom = (d_xt[1:]**2).sum(axis=2)

        # Omega
        bound  = self.bound
        Omega  = (T.switch(T.ge(denom, bound), num/denom, 1) - 1)**2
        nelems = T.mean(T.ge(denom, bound), axis=1)
        Omega  = Omega.mean(axis=1).sum()/nelems.sum()

        # Gradient w.r.t Wrec
        g_Wrec = theano.grad(Omega, Wrec)
        g_Wrec = theano.clone(g_Wrec, replace=[(d_xt, g_x), (xt, x)])

        return self.lambda_Omega * g_Wrec
