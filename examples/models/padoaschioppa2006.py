"""
Economic choice task, based on

  Neurons in the orbitofrontal cortex encode economic value.
  C. Padoa-Schioppa & J. A. Assad, Nature 2006.

  http://dx.doi.org/10.1038/nature04676

"""
from __future__ import division

import numpy as np

from pyrl import tasktools

# Inputs
inputs = tasktools.to_map('FIXATION', 'L-A', 'L-B', 'R-A', 'R-B', 'N-L', 'N-R')

# Actions
actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT')

# Trial conditions
A_to_B       = 2
juices       = [('A', 'B'), ('B', 'A')]
offers       = [(0, 1), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (2, 0)]
n_conditions = len(juices) * len(offers)

# Training
n_gradient   = n_conditions
n_validation = 50*n_conditions

# Durations
fixation     = 750
offer_on_min = 1000
offer_on_max = 2000
decision     = 750
tmax         = fixation + offer_on_max + decision

# Rewards
R_ABORTED = -1
R_B       = 0.2
R_A       = A_to_B * R_B

# Input scaling
def scale(x):
    return x/5

def get_condition(rng, dt, context={}):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    offer_on = context.get('offer-on')
    if offer_on is None:
        offer_on = tasktools.uniform(rng, dt, offer_on_min, offer_on_max)

    durations = {
        'fixation':    (0, fixation),
        'offer-on':    (fixation, fixation + offer_on),
        'decision':    (fixation + offer_on, tmax),
        'tmax':        tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    #-------------------------------------------------------------------------------------
    # Trial
    #-------------------------------------------------------------------------------------

    juice = context.get('juice')
    if juice is None:
        juice = tasktools.choice(rng, juices)

    offer = context.get('offer')
    if offer is None:
        offer = tasktools.choice(rng, offers)

    juiceL, juiceR = juice
    nB, nA = offer

    if juiceL == 'A':
        nL, nR = nA, nB
    else:
        nL, nR = nB, nA

    return {
        'durations': durations,
        'time':      time,
        'epochs':    epochs,
        'juice':     juice,
        'offer':     offer,
        'nL':        nL,
        'nR':        nR
        }

def get_step(rng, dt, trial, t, a):
    #-------------------------------------------------------------------------------------
    # Reward
    #-------------------------------------------------------------------------------------

    epochs = trial['epochs']
    status = {'continue': True}
    reward = 0
    if t-1 in epochs['fixation'] or t-1 in epochs['offer-on']:
        if a != actions['FIXATE']:
            status['continue'] = False
            reward = R_ABORTED
    elif t-1 in epochs['decision']:
        if a in [actions['CHOOSE-LEFT'], actions['CHOOSE-RIGHT']]:
            status['continue'] = False
            status['t_choice'] = t-1

            juiceL, juiceR = trial['juice']

            nB, nA = trial['offer']
            rA     = nA * R_A
            rB     = nB * R_B

            if juiceL == 'A':
                rL, rR = rA, rB
            else:
                rL, rR = rB, rA

            if a == actions['CHOOSE-LEFT']:
                if juiceL == 'A':
                    status['choice'] = 'A'
                else:
                    status['choice'] = 'B'
                status['correct'] = (rL >= rR)
                reward = rL
            elif a == actions['CHOOSE-RIGHT']:
                if juiceR == 'A':
                    status['choice'] = 'A'
                else:
                    status['choice'] = 'B'
                status['correct'] = (rR >= rL)
                reward = rR

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    u = np.zeros(len(inputs))
    if t not in epochs['decision']:
        u[inputs['FIXATION']] = 1
    if t in epochs['offer-on']:
        juiceL, juiceR = trial['juice']
        u[inputs['L-'+juiceL]] = 1
        u[inputs['R-'+juiceR]] = 1

        u[inputs['N-L']] = scale(trial['nL'])
        u[inputs['N-R']] = scale(trial['nR'])

    #-------------------------------------------------------------------------------------

    return u, reward, status

def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.95
