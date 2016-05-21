"""
Dynamic foraging task, based on

  Matching behavior and the representation of value in the parietal cortex.
  L. P. Sugrue, G. S. Corrado, & W. T. Newsome, Science 2004.

  http://dx.doi.org/10.1126/science.1094765

"""
from __future__ import division

import numpy as np

from pyrl import tasktools

# Inputs
inputs = tasktools.to_map('FIXATION', 'L-RED', 'L-GREEN', 'R-RED', 'R-GREEN')

# Actions
actions = tasktools.to_map('FIXATE', 'SACCADE-LEFT', 'SACCADE-RIGHT')

# Trial conditions
colors       = [('RED', 'GREEN'), ('GREEN', 'RED')]
reward_rates = [(8, 1), (6, 1), (3, 1), (1, 1)]

# Durations
fixation  = 300
delay_min = 1000
delay_max = 2000
decision  = 600
tmax      = fixation + delay_max + decision

# Rewards
R_ABORTED = -1

# Training
n_gradient   = 1
n_validation = 0

lr = 0.001
baseline_lr = lr

class ContextUpdater(object):
    def __init__(self, perf, rng):
        self.perf       = perf
        self.rng        = rng
        self.block_size = 0
        self.n_trial    = 0
        self.r_RED      = 0
        self.r_GREEN    = 0

        self.reward_rate_RED   = 0
        self.reward_rate_GREEN = 0

        self.choices = []

        self._new_block()

        self.switched_to_R = False
        self.switched_to_G = False

    def _new_block(self):
        # Block size
        while True:
            block_size = self.rng.exponential(120)
            if 100 <= block_size < 200:
                break
        self.block_size = int(block_size)
        self.n_trials   = 0

        # New reward rates
        reward_rate = tasktools.choice(self.rng, reward_rates)
        if self.rng.uniform() < 0.5:
            r_RED, r_GREEN = reward_rate
        else:
            r_GREEN, r_RED = reward_rate
        r_tot = r_RED + r_GREEN
        self.reward_rate_RED   = 0.3/r_tot * r_RED
        self.reward_rate_GREEN = 0.3/r_tot * r_GREEN

        # Assign rewards
        self.r_RED   = 1*(self.rng.uniform() < self.reward_rate_RED)
        self.r_GREEN = 1*(self.rng.uniform() < self.reward_rate_GREEN)

    def update(self, status, a):
        #print(self.n_trials, self.block_size, self.reward_rate_RED, self.r_RED, self.reward_rate_GREEN, self.r_GREEN)

        # Reward collected
        if 'decision' in status:
            if status['decision'] == 'RED':
                if self.switched_to_R:
                    self.switched_to_R = False
                else:
                    self.r_RED = 0
                self.choices.append('R')
            else:
                if self.switched_to_G:
                    self.switched_to_G = False
                else:
                    self.r_GREEN = 0
                self.choices.append('G')

        # Update number of trials
        self.n_trials += 1

        # New block
        if self.n_trials == self.block_size:
            self._new_block()

        # Assign rewards
        if self.r_RED == 0:
            self.r_RED   = 1*(self.rng.uniform() < self.reward_rate_RED)
        if self.r_GREEN == 0:
            self.r_GREEN = 1*(self.rng.uniform() < self.reward_rate_GREEN)

    def get_context(self, rng):
        if self.choices and self.choices[-1] == 'G':
            R = 0
            #self.switched_to_R = True
        else:
            R = self.r_RED

        if self.choices and self.choices[-1] == 'R':
            G = 0
            #self.switched_to_G = True
        else:
            G = self.r_GREEN

        return {
            'rate-red':     self.reward_rate_RED,
            'rate-green':   self.reward_rate_GREEN,
            'reward-red':   R,
            'reward-green': G
            }

def condition(rng, dt, context={}):
    delay = context.get('delay')
    if delay is None:
        delay = tasktools.uniform(rng, dt, delay_min, delay_max)

    durations = {
        'fixation': (0, fixation),
        'delay':    (fixation, fixation + delay),
        'decision': (fixation + delay, tmax),
        'tmax':     tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    return {
        'durations':    durations,
        'time':         time,
        'epochs':       epochs,
        'colors':       tasktools.choice(rng, colors),
        'rate-red':     context.get('rate-red'),
        'rate-green':   context.get('rate-green'),
        'reward-red':   context.get('reward-red'),
        'reward-green': context.get('reward-green')
        }

def step(rng, dt, trial, t, a):
    #-------------------------------------------------------------------------------------
    # Reward
    #-------------------------------------------------------------------------------------

    epochs = trial['epochs']
    status = {'continue': True}
    reward = 0
    if t-1 in epochs['fixation'] or t-1 in epochs['delay']:
        if a != actions['FIXATE']:
            status['continue'] = False
            reward = R_ABORTED
    elif t-1 in epochs['decision']:
        if a in [actions['SACCADE-LEFT'] or actions['SACCADE-RIGHT']]:
            status['continue'] = False

            L, R = trial['colors']
            if (a == actions['SACCADE-LEFT'] and L == 'RED'
                or a == actions['SACCADE-RIGHT'] and R == 'RED'):
                status['decision'] = 'RED'
            else:
                status['decision'] = 'GREEN'

            if L == 'RED':
                r_L = trial['reward-red']
                r_R = trial['reward-green']
            else:
                r_R = trial['reward-red']
                r_L = trial['reward-green']

            if a == actions['SACCADE-LEFT']:
                reward = r_L
            else:
                reward = r_R

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    u = np.zeros(len(inputs))
    if t not in epochs['decision']:
        u[inputs['FIXATION']] = 1
    if t in epochs['delay'] or t in epochs['decision']:
        L, R = trial['colors']
        u[inputs['L-'+L]] = 1
        u[inputs['R-'+R]] = 1

    #-------------------------------------------------------------------------------------

    return u, reward, status

class Performance(object):
    def __init__(self):
        self.choiceR   = []
        self.choiceG   = []
        self.rateR     = []
        self.rateG     = []
        self.decisions = []
        self.rewardR   = []
        self.rewardG   = []

    def update(self, trial, status):
        self.decisions.append('decision' in status)

        if self.decisions[-1]:
            self.rateR.append(trial['rate-red'])
            self.rateG.append(trial['rate-green'])
            self.rewardR.append(trial['reward-red'])
            self.rewardG.append(trial['reward-green'])
            if status['decision'] == 'RED':
                self.choiceR.append(1)
                self.choiceG.append(0)
            else:
                self.choiceR.append(0)
                self.choiceG.append(1)

    @property
    def n_trials(self):
        return len(self.decisions)

    @property
    def n_decisions(self):
        return sum(self.decisions)

    def display(self):
        n_trials    = self.n_trials
        n_decisions = self.n_decisions
        print("  Prop. decision: {}/{} = {:.3f}"
              .format(n_decisions, n_trials, n_decisions/n_trials))
