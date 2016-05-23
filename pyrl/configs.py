import numpy as np

required = ['inputs', 'actions', 'tmax', 'n_gradient', 'n_validation']
default  = {
    'Performance':   None,
    'N':             50,
    'lr':            0.005,
    'baseline_lr':   0.005,
    'max_iter':      1000000,
    'fix':           [],
    'target_reward': np.inf,
    'mode':          'episodic',
    'network_type':  'gru',
    'dt':            10,
    'tau':           100,
    'var_rec':       0.02,
    'checkfreq':     50,
    'L2_r':          0.002,
    'baseline_L2_r': 0,
    'L1_Wrec':       0,
    'L2_Wrec':       0,
    'R_ABORTED':     -1,
    'policy_seed':   1,
    'baseline_seed': 2
    }
