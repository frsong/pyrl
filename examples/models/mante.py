"""
Context-dependent integration task, based on

  Context-dependent computation by recurrent dynamics in prefrontal cortex.
  V. Mante, D. Sussillo, K. V. Shinoy, & W. T. Newsome, Nature 2013.

  http://dx.doi.org/10.1038/nature12742

"""
from __future__ import division

import numpy as np

from pyrl import tasktools

# Inputs
inputs = tasktools.to_map('FIXATION', 'MOTION', 'COLOR',
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
n_validation = 50*n_conditions

# Time step
#dt = 10

#max_iter = 650

# Input noise
sigma = np.sqrt(2*100*0.02)

# Rewards
R_ABORTED = -1
R_CORRECT = +1

# Epoch durations
fixation_min = 250
fixation_max = 750
stimulus     = 750
delay_min    = 300
delay_mean   = 300
delay_max    = 1200
decision     = 500
tmax         = fixation_max + stimulus + delay_min + delay_max + decision

def get_condition(rng, dt, context={}):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    fixation = context.get('fixation')
    if fixation is None:
        fixation = tasktools.uniform(rng, dt, fixation_min, fixation_max)

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

    contexts_      = context.get('context',      contexts)
    left_rights_m_ = context.get('left_right_m', left_rights)
    left_rights_c_ = context.get('left_right_c', left_rights)
    cohs_m_        = context.get('coh_m',        cohs)
    cohs_c_        = context.get('coh_c',        cohs)

    return {
        'durations':    durations,
        'time':         time,
        'epochs':       epochs,
        'context':      rng.choice(contexts_),
        'left_right_m': rng.choice(left_rights_m_),
        'left_right_c': rng.choice(left_rights_c_),
        'coh_m':        rng.choice(cohs_m_),
        'coh_c':        rng.choice(cohs_c_)
        }

# Input scaling
def scale(coh):
    return (1 + coh/100)/2

def get_step(rng, dt, trial, t, a):
    #-------------------------------------------------------------------------------------
    # Reward
    #-------------------------------------------------------------------------------------

    time   = trial['time']
    epochs = trial['epochs']
    status = {'continue': True}
    reward = 0

    if t-1 in epochs['fixation'] or t-1 in epochs['stimulus'] or t-1 in epochs['delay']:
        if a != actions['FIXATE']:
            status['continue'] = False
            reward = R_ABORTED
    elif t-1 in epochs['decision']:
        if a == actions['CHOOSE-LEFT']:
            status['continue'] = False
            status['choice']   = 'L'
            status['t_choice'] = time[t-1]
            if trial['context'] == 'm':
                if trial['left_right_m'] < 0:
                    status['correct'] = True
                    reward = R_CORRECT
                else:
                    status['correct'] = False
            else:
                if trial['left_right_c'] < 0:
                    status['correct'] = True
                    reward = R_CORRECT
                else:
                    status['correct'] = False
        elif a == actions['CHOOSE-RIGHT']:
            status['continue'] = False
            status['choice']   = 'R'
            status['t_choice'] = time[t-1]
            if trial['context'] == 'm':
                if trial['left_right_m'] > 0:
                    status['correct'] = True
                    reward = R_CORRECT
                else:
                    status['correct'] = False
            else:
                if trial['left_right_c'] > 0:
                    status['correct'] = True
                    reward = R_CORRECT
                else:
                    status['correct'] = False

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
        u[inputs['FIXATION']] = 1
        u[context]            = 1
    if t in epochs['stimulus']:
        u[high_m] = scale(+trial['coh_m']) + rng.normal(scale=sigma)/np.sqrt(dt)
        u[low_m]  = scale(-trial['coh_m']) + rng.normal(scale=sigma)/np.sqrt(dt)
        u[high_c] = scale(+trial['coh_c']) + rng.normal(scale=sigma)/np.sqrt(dt)
        u[low_c]  = scale(-trial['coh_c']) + rng.normal(scale=sigma)/np.sqrt(dt)

    #-------------------------------------------------------------------------------------

    return u, reward, status

def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.9
