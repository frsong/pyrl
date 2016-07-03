"""
Context-dependent integration task, based on

  Context-dependent computation by recurrent dynamics in prefrontal cortex.
  V Mante, D Sussillo, KV Shinoy, & WT Newsome, Nature 2013.

  http://dx.doi.org/10.1038/nature12742

"""
from __future__ import division

import numpy as np

from pyrl import tasktools

# Inputs
inputs = tasktools.to_map('MOTION', 'COLOR',
                          'MOTION-LEFT', 'MOTION-RIGHT',
                          'COLOR-LEFT', 'COLOR-RIGHT')

# Actions
actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT')

# Trial conditions
contexts     = ['m', 'c']
left_rights  = [-1, 1]
cohs         = [5, 15, 50]
n_conditions = len(contexts) * (len(left_rights)*len(cohs))**2

# Training
n_gradient   = n_conditions
n_validation = 100*n_conditions

# Input noise
sigma = np.sqrt(2*100*0.025)

# Rewards
R_ABORTED = -1
R_CORRECT = +1

# Epoch durations
fixation   = 750
stimulus   = 750
delay_min  = 300
delay_mean = 300
delay_max  = 1200
decision   = 500
tmax       = fixation + stimulus + delay_min + delay_max + decision

def get_condition(rng, dt, context={}):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    delay = context.get('delay')
    if delay is None:
        delay = delay_min + tasktools.truncated_exponential(rng, dt, delay_mean,
                                                            xmax=delay_max)

    durations = {
        'fixation':  (0, fixation),
        'stimulus':  (fixation, fixation + stimulus),
        'delay':     (fixation + stimulus, fixation + stimulus + delay),
        'decision':  (fixation + stimulus + delay, tmax),
        'tmax':      tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    #-------------------------------------------------------------------------------------
    # Trial
    #-------------------------------------------------------------------------------------

    context_ = context.get('context')
    if context_ is None:
        context_ = rng.choice(contexts)

    left_right_m = context.get('left_right_m')
    if left_right_m is None:
        left_right_m = rng.choice(left_rights)

    left_right_c = context.get('left_right_c')
    if left_right_c is None:
        left_right_c = rng.choice(left_rights)

    coh_m = context.get('coh_m')
    if coh_m is None:
        coh_m = rng.choice(cohs)

    coh_c = context.get('coh_c')
    if coh_c is None:
        coh_c = rng.choice(cohs)

    return {
        'durations':    durations,
        'time':         time,
        'epochs':       epochs,
        'context':      context_,
        'left_right_m': left_right_m,
        'left_right_c': left_right_c,
        'coh_m':        coh_m,
        'coh_c':        coh_c
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
    if t-1 not in epochs['decision']:
        if a != actions['FIXATE']:
            status['continue'] = False
            reward = R_ABORTED
    elif t-1 in epochs['decision']:
        if a == actions['CHOOSE-LEFT']:
            status['continue'] = False
            status['choice']   = 'L'
            status['t_choice'] = t-1
            if trial['context'] == 'm':
                status['correct'] = (trial['left_right_m'] < 0)
            else:
                status['correct'] = (trial['left_right_c'] < 0)
            if status['correct']:
                reward = R_CORRECT
        elif a == actions['CHOOSE-RIGHT']:
            status['continue'] = False
            status['choice']   = 'R'
            status['t_choice'] = t-1
            if trial['context'] == 'm':
                status['correct'] = (trial['left_right_m'] > 0)
            else:
                status['correct'] = (trial['left_right_c'] > 0)
            if status['correct']:
                reward = R_CORRECT

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    if trial['context'] == 'm':
        context = inputs['MOTION']
    else:
        context = inputs['COLOR']

    if trial['left_right_m'] < 0:
        high_m = inputs['MOTION-LEFT']
        low_m  = inputs['MOTION-RIGHT']
    else:
        high_m = inputs['MOTION-RIGHT']
        low_m  = inputs['MOTION-LEFT']

    if trial['left_right_c'] < 0:
        high_c = inputs['COLOR-LEFT']
        low_c  = inputs['COLOR-RIGHT']
    else:
        high_c = inputs['COLOR-RIGHT']
        low_c  = inputs['COLOR-LEFT']

    u = np.zeros(len(inputs))
    if t in epochs['fixation'] or t in epochs['stimulus'] or t in epochs['delay']:
        u[context] = 1
    if t in epochs['stimulus']:
        u[high_m] = scale(+trial['coh_m']) + rng.normal(scale=sigma)/np.sqrt(dt)
        u[low_m]  = scale(-trial['coh_m']) + rng.normal(scale=sigma)/np.sqrt(dt)
        u[high_c] = scale(+trial['coh_c']) + rng.normal(scale=sigma)/np.sqrt(dt)
        u[low_c]  = scale(-trial['coh_c']) + rng.normal(scale=sigma)/np.sqrt(dt)

    #-------------------------------------------------------------------------------------

    return u, reward, status

def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.85
