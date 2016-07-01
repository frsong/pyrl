"""
Perceptual decision-making, reaction-time version.

"""
from __future__ import division

import numpy as np

from pyrl import tasktools

# Inputs
inputs = tasktools.to_map('FIXATION', 'LEFT', 'RIGHT')

# Actions
actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT')

# Trial conditions
left_rights  = [-1, 1]
cohs         = [0, 3.2, 6.4, 12.8, 25.6, 51.2]
n_conditions = len(left_rights)*len(cohs)

# Training
n_gradient   = n_conditions
n_validation = 100*n_conditions

# Input noise
sigma = np.sqrt(2*100*0.013)

# Durations
fixation = 750
stimulus = 2000
tmax     = fixation + stimulus

# Rewards
R_ABORTED = -1
R_CORRECT = +1

def get_condition(rng, dt, context={}):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    durations = {
        'fixation': (0, fixation),
        'stimulus': (fixation, fixation + stimulus),
        'tmax':     tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    #-------------------------------------------------------------------------------------
    # Trial
    #-------------------------------------------------------------------------------------

    left_right = context.get('left_right')
    if left_right is None:
        left_right = rng.choice(left_rights)

    coh = context.get('coh')
    if coh is None:
        coh = rng.choice(cohs)

    return {
        'durations':   durations,
        'time':        time,
        'epochs':      epochs,
        'left_right':  left_right,
        'coh':         coh
        }

# Input scaling
def scale(coh):
    return (1 + coh/100)/2

def get_step(rng, dt, trial, t, a):
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
    elif t-1 in epochs['stimulus']:
        if a == actions['CHOOSE-LEFT']:
            status['continue'] = False
            status['choice']   = 'L'
            status['t_choice'] = t-1
            status['correct']  = (trial['left_right'] < 0)
            if status['correct']:
                reward = R_CORRECT
        elif a == actions['CHOOSE-RIGHT']:
            status['continue'] = False
            status['choice']   = 'R'
            status['t_choice'] = t-1
            status['correct']  = (trial['left_right'] > 0)
            if status['correct']:
                reward = R_CORRECT

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    if trial['left_right'] < 0:
        high = inputs['LEFT']
        low  = inputs['RIGHT']
    else:
        high = inputs['RIGHT']
        low  = inputs['LEFT']

    u = np.zeros(len(inputs))
    if t in epochs['fixation']:
        u[inputs['FIXATION']] = 1
    if t in epochs['stimulus']:
        u[high] = scale(+trial['coh']) + rng.normal(scale=sigma)/np.sqrt(dt)
        u[low]  = scale(-trial['coh']) + rng.normal(scale=sigma)/np.sqrt(dt)

    #-------------------------------------------------------------------------------------

    return u, reward, status

def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.8
