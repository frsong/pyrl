"""
Perceptual decision-making task, based on

  Bounded integration in parietal cortex underlies decisions even when viewing
  duration is dictated by the environment.
  R. Kiani, T. D. Hanks, & M. N. Shadlen, JNS 2008.

  http://dx.doi.org/10.1523/JNEUROSCI.4761-07.2008

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
cohs         = [0, 6.4, 12.8, 25.6, 51.2]
n_conditions = len(left_rights)*len(cohs)

# Training
n_gradient   = n_conditions
n_validation = 100*n_conditions

# Input noise
sigma = 1

#var_rec = 2.5

# Durations
fixation          = 500
stimulus_min      = 80
stimulus_mean     = 330
stimulus_max      = 1500
choice_initiation = 500
hold              = 50
decision          = choice_initiation + hold
tmax              = fixation + stimulus_max + decision

# Rewards
R_ABORTED = -1
R_CORRECT = +1

# Input scaling
def scale(coh):
    return (1 + coh/100)/2

class Task(object):
    def start_trial(self):
        self.choice   = None
        self.t_choice = None
        self.correct  = None

    def get_condition(self, rng, dt, context={}):
        #---------------------------------------------------------------------------------
        # Epochs
        #---------------------------------------------------------------------------------

        stimulus = context.get(
            'stimulus',
            tasktools.truncated_exponential(rng, dt, stimulus_mean,
                                            xmin=stimulus_min, xmax=stimulus_max)
            )

        durations = {
            'fixation':          (0, fixation),
            'stimulus':          (fixation, fixation + stimulus),
            'choice_initiation': (fixation + stimulus,
                                  fixation + stimulus + choice_initiation),
            'decision':          (fixation + stimulus, fixation + stimulus + decision),
            'tmax':              tmax
            }
        time, epochs = tasktools.get_epochs_idx(dt, durations)

        #---------------------------------------------------------------------------------
        # Trial
        #---------------------------------------------------------------------------------

        left_rights_ = context.get('left_rights', left_rights)
        cohs_        = context.get('cohs', cohs)

        return {
            'durations':   durations,
            'time':        time,
            'epochs':      epochs,
            'left_right':  rng.choice(left_rights_),
            'coh':         rng.choice(cohs_)
            }

    def get_step(self, rng, dt, trial, t, a):
        #---------------------------------------------------------------------------------
        # Reward
        #---------------------------------------------------------------------------------

        time   = trial['time']
        epochs = trial['epochs']
        status = {'continue': True}
        reward = 0

        if t-1 in epochs['fixation'] or t-1 in epochs['stimulus']:
            if a != actions['FIXATE']:
                status['continue'] = False
                reward = R_ABORTED
        elif t-1 in epochs['decision']:
            if self.choice is None:
                if t-1 not in epochs['choice_initiation']:
                    status['continue'] = False
                    reward = R_ABORTED
                elif a in [actions['CHOOSE-LEFT'], actions['CHOOSE-RIGHT']]:
                    self.choice   = a
                    self.t_choice = time[t-1]
                    if a == actions['CHOOSE-LEFT']:
                        if trial['coh'] == 0:
                            self.correct = tasktools.choice(rng, [True, False])
                        elif trial['left_right'] < 0:
                            self.correct = True
                        elif trial['left_right'] > 0:
                            self.correct = False
                    elif a == actions['CHOOSE-RIGHT']:
                        if trial['coh'] == 0:
                            self.correct = tasktools.choice(rng, [True, False])
                        elif trial['left_right'] > 0:
                            self.correct = True
                        elif trial['left_right'] < 0:
                            self.correct = False
            else:
                if a == self.choice:
                    if time[t-1] - self.t_choice >= hold:
                        status['continue'] = False
                        status['choice']   = self.choice
                        status['t_choice'] = self.t_choice
                        status['correct']  = self.correct
                        if self.correct:
                            reward = R_CORRECT
                else:
                    status['continue'] = False
                    reward = R_ABORTED

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
        if t in epochs['fixation'] or t in epochs['stimulus']:
            u[inputs['FIXATION']] = 1
        if t in epochs['stimulus']:
            u[high] = scale(+trial['coh']) + rng.normal(scale=sigma)/np.sqrt(dt)
            u[low]  = scale(-trial['coh']) + rng.normal(scale=sigma)/np.sqrt(dt)

        #---------------------------------------------------------------------------------

        return u, reward, status

    def terminate(self, perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.8
