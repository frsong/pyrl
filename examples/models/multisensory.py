"""
Multisensory integration, based on

  A category-free neural population supports evolving demands during decision-making.
  D Raposo, MT Kaufman, & AK Churchland, Nature Neurosci. 2014.

  http://dx.doi.org/10.1038/nn.3865

"""
from __future__ import division

import numpy as np

from pyrl import tasktools

# Inputs
inputs = tasktools.to_map('FIXATION', 'VISUAL-P', 'VISUAL-N', 'AUDITORY-P', 'AUDITORY-N')

# Actions
actions = tasktools.to_map('FIXATE', 'CHOOSE-LOW', 'CHOOSE-HIGH')

# Trial conditions
mods         = ['v', 'a', 'va']
freqs        = range(9, 16+1)
n_conditions = len(mods) * len(freqs)

# Discrimination boundary
boundary = 12.5

# Training
n_gradient   = n_conditions
n_validation = 100*n_conditions

# Input noise
sigma = np.sqrt(2*100*0.017)

# Separate visual and auditory inputs
N    = 100
Wins = []
for i in xrange(3):
    Win = np.zeros((len(inputs), N))
    Win[inputs['FIXATION']]          = 1
    Win[inputs['VISUAL-P'],:N//3]    = 1
    Win[inputs['VISUAL-N'],:N//3]    = 1
    Win[inputs['AUDITORY-P'],-N//3:] = 1
    Win[inputs['AUDITORY-N'],-N//3:] = 1
    Wins.append(Win)
Win      = np.concatenate(Wins, axis=1)
Win_mask = Win.copy()

# Epoch durations
fixation = 750
stimulus = 1000
decision = 500
tmax     = fixation + stimulus + decision

def get_condition(rng, dt, context={}):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    durations = {
        'fixation': (0, fixation),
        'stimulus': (fixation, fixation + stimulus),
        'decision': (fixation + stimulus, tmax),
        'tmax':     tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    #-------------------------------------------------------------------------------------

    mod = context.get('mod')
    if mod is None:
        mod = rng.choice(mods)

    freq = context.get('freq')
    if freq is None:
        freq = rng.choice(freqs)

    return {
        'durations': durations,
        'time':      time,
        'epochs':    epochs,
        'mod':       mod,
        'freq':      freq
        }

# Rewards
R_ABORTED = -1
R_CORRECT = +1

# Input scaling
fmin = 0
fmax = 2*boundary

def scale(f):
    return (f - fmin)/(fmax - fmin)

def scale_p(f):
    return (1 + scale(f))/2

def scale_n(f):
    return (1 - scale(f))/2

def get_step(rng, dt, trial, t, a):
    #-------------------------------------------------------------------------------------
    # Reward
    #-------------------------------------------------------------------------------------

    epochs = trial['epochs']
    status = {'continue': True}
    reward = 0
    if t-1 not in epochs['decision']:
        if a != actions['FIXATE']:
            status['continue'] = False
            reward = R_ABORTED
    elif t-1 in epochs['decision']:
        if a == actions['CHOOSE-LOW']:
            status['continue'] = False
            status['choice']   = 'L'
            status['t_choice'] = t-1
            status['correct'] = (trial['freq'] < boundary)
            if status['correct']:
                reward = R_CORRECT
        elif a == actions['CHOOSE-HIGH']:
            status['continue'] = False
            status['choice']   = 'H'
            status['t_choice'] = t-1
            status['correct'] = (trial['freq'] > boundary)
            if status['correct']:
                reward = R_CORRECT

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    u = np.zeros(len(inputs))
    if t in epochs['fixation'] or t in epochs['stimulus']:
        u[inputs['FIXATION']] = 1
    if t in epochs['stimulus']:
        if 'v' in trial['mod']:
            u[inputs['VISUAL-P']] = (scale_p(trial['freq'])
                                     + rng.normal(scale=sigma)/np.sqrt(dt))
            u[inputs['VISUAL-N']] = (scale_n(trial['freq'])
                                     + rng.normal(scale=sigma)/np.sqrt(dt))
        if 'a' in trial['mod']:
            u[inputs['AUDITORY-P']] = (scale_p(trial['freq'])
                                       + rng.normal(scale=sigma)/np.sqrt(dt))
            u[inputs['AUDITORY-N']] = (scale_n(trial['freq'])
                                       + rng.normal(scale=sigma)/np.sqrt(dt))

    #-------------------------------------------------------------------------------------

    return u, reward, status

def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.8
