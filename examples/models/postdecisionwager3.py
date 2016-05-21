"""
Perceptual decision-making with postdecision wagering, based on

  Representation of confidence associated with a decision by
  neurons in the parietal cortex.
  R. Kiani & M. N. Shadlen, Science 2009.

  http://dx.doi.org/10.1126/science.1169405

"""
from __future__ import division

import numpy as np

from pyrl import tasktools

# Inputs
inputs = tasktools.to_map('FIXATION', 'LEFT', 'RIGHT', 'SURE')

# Actions
actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT', 'CHOOSE-SURE')

# Trial conditions
wagers       = [True, False]
left_rights  = [-1, 1]
cohs         = [0, 3.2, 6.4, 12.8, 25.6, 51.2]
n_conditions = len(wagers) * len(left_rights) * len(cohs)

# Training
n_gradient   = n_conditions
n_validation = 100*n_conditions

dt = 10

# Input noise
#sigma = np.sqrt(2*100*0.007)
#sigma = np.sqrt(2*100*0.01)
sigma = np.sqrt(2*100*0.0072)

var_rec = 0.02

# Durations
fixation_min  = 250
fixation_max  = 750
stimulus_min  = 100
stimulus_mean = 180
stimulus_max  = 800
delay_min     = 1200
delay_mean    = 1350
delay_max     = 1800
sure_min      = 500
sure_mean     = 575
sure_max      = 750
decision      = 500
tmax          = fixation_max + stimulus_max + delay_max + decision

# Rewards
R_ABORTED = -1
R_CORRECT = +1
R_SURE    = 0.7*R_CORRECT

#lr = 0.004
#baseline_lr = 0.004

def get_condition(rng, dt, context={}):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    fixation = context.get('fixation')
    if fixation is None:
        fixation = tasktools.uniform(rng, dt, fixation_min, fixation_max)

    stimulus = context.get('stimulus')
    if stimulus is None:
        stimulus = stimulus_min + tasktools.truncated_exponential(rng, dt, stimulus_mean,
                                                                  xmax=stimulus_max)

    delay = context.get('delay')
    if delay is None:
        delay = tasktools.truncated_exponential(rng, dt, delay_mean,
                                                xmin=delay_min, xmax=delay_max)

    sure_onset = context.get('sure_onset')
    if sure_onset is None:
        sure_onset = tasktools.truncated_exponential(rng, dt, sure_mean,
                                                     xmin=sure_min, xmax=sure_max)

    durations = {
        'fixation':  (0, fixation),
        'stimulus':  (fixation, fixation + stimulus),
        'delay':     (fixation + stimulus, fixation + stimulus + delay),
        'sure':      (fixation + stimulus + sure_onset, tmax),
        'decision':  (fixation + stimulus + delay, tmax),
        'tmax':      tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    #-------------------------------------------------------------------------------------
    # Trial
    #-------------------------------------------------------------------------------------

    wager = context.get('wager')
    if wager is None:
        wager = rng.choice(wagers)

    coh = context.get('coh')
    if coh is None:
        coh = rng.choice(cohs)

    left_right = context.get('left_right')
    if left_right is None:
        left_right = rng.choice(left_rights)

    return {
        'durations':  durations,
        'time':       time,
        'epochs':     epochs,
        'wager':      wager,
        'left_right': left_right,
        'coh':        coh
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
            status['correct']  = (trial['left_right'] < 0)
            if status['correct']:
                reward = R_CORRECT
        elif a == actions['CHOOSE-RIGHT']:
            status['continue'] = False
            status['choice']   = 'R'
            status['t_choice'] = time[t-1]
            status['correct']  = (trial['left_right'] > 0)
            if status['correct']:
                reward = R_CORRECT
        elif a == actions['CHOOSE-SURE']:
            status['continue'] = False
            if trial['wager']:
                status['choice']   = 'S'
                status['t_choice'] = time[t-1]
                reward = R_SURE
            else:
                reward = R_ABORTED

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
    if t in epochs['fixation'] or t in epochs['stimulus'] or t in epochs['delay']:
        u[inputs['FIXATION']] = 1
    if t in epochs['stimulus']:
        u[high] = scale(+trial['coh']) + rng.normal(scale=sigma)/np.sqrt(dt)
        u[low]  = scale(-trial['coh']) + rng.normal(scale=sigma)/np.sqrt(dt)
    if trial['wager'] and t in epochs['sure']:
        u[inputs['SURE']] = 1

    #-------------------------------------------------------------------------------------

    return u, reward, status

class Performance(object):
    def __init__(self):
        self.wagers    = []
        self.corrects  = []
        self.choices   = []
        self.t_choices = []

    def update(self, trial, status):
        self.wagers.append(trial['wager'])
        self.corrects.append(status.get('correct', None))
        self.choices.append(status.get('choice', None))
        self.t_choices.append(status.get('t_choice', None))

    @property
    def n_correct(self):
        return sum([c for c in self.corrects if c is not None])

    @property
    def n_sure_decision(self):
        return len([1 for w, c in zip(self.wagers, self.choices) if w and c is not None])

    @property
    def n_trials(self):
        return len(self.choices)

    @property
    def n_decision(self):
        return len([1 for c in self.choices if c in ['L', 'R']])

    @property
    def n_sure(self):
        return len([1 for c in self.choices if c == 'S'])

    @property
    def n_answer(self):
        return len([1 for c in self.choices if c is not None])

    def display(self):
        n_trials        = self.n_trials
        n_decision      = self.n_decision
        n_correct       = self.n_correct
        n_sure_decision = self.n_sure_decision
        n_sure          = self.n_sure
        n_answer        = self.n_answer
        print("  Prop. answer:   {}/{} = {:.3f}"
              .format(n_answer, n_trials, n_answer/n_trials))
        print("  Prop. decision: {}/{} = {:.3f}"
              .format(n_decision, n_trials, n_decision/n_trials))
        if n_decision > 0:
            print("  Prop. correct:  {}/{} = {:.3f}"
                  .format(n_correct, n_decision, n_correct/n_decision))
        print("  Prop. wager:    {}/{} = {:.3f}"
              .format(sum(self.wagers), n_trials, sum(self.wagers)/n_trials))
        if n_sure_decision > 0:
            print("  Prop. sure:     {}/{} = {:.3f}"
                  .format(n_sure, n_sure_decision, n_sure/n_sure_decision))

def terminate(perf):
    p_answer  = perf.n_answer/perf.n_trials
    p_correct = tasktools.divide(perf.n_correct, perf.n_decision)

    return p_answer >= 0.99 and p_correct >= 0.85
