from __future__ import division

import numpy as np
from   scipy import stats

from pyrl import tasktools

#/////////////////////////////////////////////////////////////////////////////////////////

# Inputs
inputs = ['FIXATION']
inputs = {o: i for i, o in enumerate(inputs)}

# Actions
actions = ['FIXATE', 'SACCADE_LEFT', 'SACCADE_RIGHT']
actions = {a: i for i, a in enumerate(actions)}

# Trial conditions
left_rights = ['L', 'R']

# Time step (ms)
dt = 250

iti       = 500
fixation  = 500
decision  = 500
terminate = 250
tmax      = iti + fixation + decision + terminate

Tmax = int(tmax/dt)

n_max = 5

class ContextUpdater(object):
    def __init__(self, perf, verbose=False):
        self.verbose = verbose

        # Action at time t-n
        self.a_tmn = {n: None for n in xrange(n_max)}

        # Number of times saccading right conditional on saccading left or right at time t-n
        self.R_tmn = {n: {'L': 0, 'R': 0} for n in xrange(n_max)}

        # Number of times saccading conditional on saccading at time t-n
        self.N_tmn = {n: {'L': 0, 'R': 0} for n in xrange(n_max)}

    def update(self, status, a):
        if 'correct' in status:
            # Update counts
            for n in xrange(n_max):
                if self.a_tmn[n] == actions['SACCADE_LEFT']:
                    self.N_tmn[n]['L'] += 1
                    if a == actions['SACCADE_RIGHT']:
                        self.R_tmn[n]['L'] += 1
                elif self.a_tmn[n] == actions['SACCADE_RIGHT']:
                    self.N_tmn[n]['R'] += 1
                    if a == actions['SACCADE_RIGHT']:
                        self.R_tmn[n]['R'] += 1

            # Update history
            for n in xrange(1, n_max):
                self.a_tmn[n] = self.a_tmn[n-1]
            self.a_tmn[0] = a

    def get_context(self, rng):
        p = [0.5, 0.5]

        psig = 0.05

        pR_sig = []
        for n in xrange(n_max):
            if self.a_tmn[n] == actions['SACCADE_LEFT']:
                if self.N_tmn[n]['L'] > 0:
                    pval = stats.binom_test(self.R_tmn[n]['L'], self.N_tmn[n]['L'])
                    if pval < psig:
                        pR = self.R_tmn[n]['L']/self.N_tmn[n]['L']
                        pR_sig.append(pR)
            elif self.a_tmn[n] == actions['SACCADE_RIGHT']:
                if self.N_tmn[n]['R'] > 0:
                    pval = stats.binom_test(self.R_tmn[n]['R'], self.N_tmn[n]['R'])
                    if pval < psig:
                        pR = self.R_tmn[n]['R']/self.N_tmn[n]['R']
                        pR_sig.append(pR)
        if pR_sig:
            idx = np.argmax(abs(np.array(pR_sig) - 0.5))
            pR  = pR_sig[idx]
            p   = [pR, 1-pR]

        if self.verbose:
            if p[0] > 0.5:
                print("pR = {:.2f}: Biased toward R, compensating toward L.".format(p[0]))
            elif p[0] < 0.5:
                print("pR = {:.2f}: Biased toward L, compensating toward R.".format(p[0]))

        left_right = rng.choice(left_rights, p=p)

        return {'left_rights': [left_right]}

def generate_trial_condition(rng, context={}):
    """
    Generate trial condition.

    Algorithm 1.

    """
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    epoch_durations = {
        'iti':       (0, iti),
        'fixation':  (iti, iti + fixation),
        'decision':  (iti + fixation, iti + fixation + decision),
        'terminate': (iti + fixation + decision, tmax),
        'tmax':      tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, epoch_durations)

    #-------------------------------------------------------------------------------------
    # Trial
    #-------------------------------------------------------------------------------------

    left_rights_ = context.get('left_rights', left_rights)

    trial = {
        'epoch_durations': epoch_durations,
        'time':            time,
        'epochs':          epochs,
        'left_right':      rng.choice(left_rights_)
        }

    #-------------------------------------------------------------------------------------

    return trial

# Rewards
R_ABORTED   = -1
R_CORRECT   = +1
R_INCORRECT = 0
R_TERMINATE = -1

# Interactive trial
def step(rng, trial, t, a):
    """
    Parameters
    ----------

    t: `t > 0`

    Returns
    -------

    u :      input at time `t`
    reward : reward at time `t` given the action determined at time `t-1`
    status : trial status

    """
    #-------------------------------------------------------------------------------------
    # Reward
    #-------------------------------------------------------------------------------------

    epochs = trial['epochs']
    status = {'continue': True}
    reward = 0
    if t-1 in epochs['fixation']:
        if a != actions['FIXATE']:
            status['continue'] = False
            reward = R_ABORTED
    elif t-1 in epochs['decision']:
        if a == actions['SACCADE_LEFT']:
            if trial['left_right'] == 'L':
                status['continue'] = False
                status['correct']  = True
                reward = R_CORRECT
            elif trial['left_right'] == 'R':
                status['continue'] = False
                status['correct']  = False
                reward = R_INCORRECT
        elif a == actions['SACCADE_RIGHT']:
            if trial['left_right'] == 'R':
                status['continue'] = False
                status['correct']  = True
                reward = R_CORRECT
            elif trial['left_right'] == 'L':
                status['continue'] = False
                status['correct']  = False
                reward = R_INCORRECT
    elif t-1 in epochs['terminate']:
        status['continue'] = False
        reward = R_TERMINATE

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    u = np.zeros(len(inputs))
    if t in epochs['fixation']:
        u[inputs['FIXATION']] = 1

    #-------------------------------------------------------------------------------------

    return u, reward, status

def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.45

#/////////////////////////////////////////////////////////////////////////////////////////

config = {
    'Nin':          inputs,
    'Nout':         actions,
    'Tmax':         Tmax,
    'fix':          ['states_0'],
    'mode':         'continuous',
    'n_gradient':   200,
    'n_validation': 2000,
    'checkfreq':    100
    }
