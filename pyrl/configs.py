import numpy as np

required = ['inputs', 'actions', 'tmax', 'n_gradient', 'n_validation']
default  = {
    'Performance':      None,
    'N':                100,
    'p0':               0.1,
    'lr':               0.005,
    'baseline_lr':      0.005,
    'max_iter':         100000,
    'fix':              [],
    'baseline_fix':     [],
    'target_reward':    np.inf,
    'mode':             'episodic',
    'network_type':     'gru',
    'R_ABORTED':        -1,
    'checkfreq':        50,
    'dt':               10,
    'tau':              100,
    'var_rec':          0.02,
    'baseline_var_rec': 0.01,
    'L2_r':             0,
    'baseline_L2_r':    0,
    'rho':              2,
    'baseline_rho':     2,
    'L1_Wrec':          0,
    'L2_Wrec':          0,
    'policy_seed':      1,
    'baseline_seed':    2
    }
