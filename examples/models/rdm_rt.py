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
#actions = tasktools.to_map('LOCATION')

# Trial conditions
left_rights  = [-1, 1]
cohs         = [0, 3.2, 6.4, 12.8, 25.6, 51.2]
n_conditions = len(left_rights)*len(cohs)

# Training
n_gradient   = n_conditions
n_validation = 100*n_conditions

#lr = 0.002
#baseline_lr = 0.002

# Input noise
sigma = np.sqrt(2*100*0.005)

# Epoch durations
fixation_min = 250
fixation_max = 750
stimulus     = 2000
choice_hold  = 50
tmax         = fixation_max + stimulus

# Rewards
R_ABORTED = -1
R_CORRECT = +1

# Input scaling
def scale(coh):
    return (1 + coh/100)/2

x_target = 0.5
r_target = 0.2

def is_fixate(x):
    return abs(x) < r_target

def is_left(x):
    return abs(x - -x_target) < r_target

def is_right(x):
    return abs(x - +x_target) < r_target

L2_r = 0

class Task(object):
    def start_trial(self):
        self.choice   = None
        self.t_choice = None
        self.correct  = None

    def get_condition(self, rng, dt, context={}):
        #---------------------------------------------------------------------------------
        # Epochs
        #---------------------------------------------------------------------------------

        fixation = context.get('fixation')
        if fixation is None:
            fixation = tasktools.uniform(rng, dt, fixation_min, fixation_max)

        durations = {
            'fixation':  (0, fixation),
            'stimulus':  (fixation, fixation + stimulus),
            'tmax':      tmax
            }
        time, epochs = tasktools.get_epochs_idx(dt, durations)

        #---------------------------------------------------------------------------------
        # Trial
        #---------------------------------------------------------------------------------

        left_rights_ = context.get('left_rights', left_rights)
        cohs_        = context.get('cohs', cohs)

        return {
            'durations':  durations,
            'time':       time,
            'epochs':     epochs,
            'left_right': rng.choice(left_rights_),
            'coh':        rng.choice(cohs_)
        }

    def get_step(self, rng, dt, trial, t, a):
        #---------------------------------------------------------------------------------
        # Reward
        #---------------------------------------------------------------------------------

        epochs = trial['epochs']
        status = {'continue': True}
        reward = 0

        if t-1 in epochs['fixation']:
            if a != actions['FIXATE']:
            #if not is_fixate(a):
                status['continue'] = False
                reward = R_ABORTED
        elif t-1 in epochs['stimulus']:
            if self.choice is None:
                #if is_fixate(a):
                #    pass
                if a in [actions['CHOOSE-LEFT'], actions['CHOOSE-RIGHT']]:
                #elif is_left(a) or is_right(a):
                    self.t_choice = t-1
                    if a == actions['CHOOSE-LEFT']:
                    #if is_left(a):
                        self.choice  = 'L'
                        self.correct = (trial['left_right'] < 0)
                    elif a == actions['CHOOSE-RIGHT']:
                    #elif is_right(a):
                        self.choice = 'R'
                        self.correct = (trial['left_right'] > 0)
                    else:
                        raise ValueError(a)

                    status['continue'] = False
                    status['choice']   = self.choice
                    status['t_choice'] = self.t_choice
                    status['correct']  = self.correct
                    if self.correct:
                        reward = R_CORRECT
                        #t_stimulus = trial['durations']['stimulus'][0]
                        #reward = np.exp(-(time[t-1] - t_stimulus)/500) * R_CORRECT
                #else:
                #    status['continue'] = False
                #    reward = R_ABORTED

        #---------------------------------------------------------------------------------
        # Inputs
        #---------------------------------------------------------------------------------

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

        #---------------------------------------------------------------------------------

        return u, reward, status

    def terminate(self, perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.9
