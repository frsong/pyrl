import imp
import os

import numpy as np

from pyrl          import utils
from pyrl.figtools import Figure

#=========================================================================================
# Files
#=========================================================================================

here   = utils.get_here(__file__)
parent = utils.get_parent(here)

# Paths
scratchpath = os.environ.get('SCRATCH')
if scratchpath is None:
    scratchpath = os.path.join(os.environ['HOME'], 'scratch')
trialspath  = os.path.join(scratchpath, 'work', 'pyrl', 'examples')
analysispath = os.path.join(parent, 'examples', 'analysis')
modelspath   = os.path.join(parent, 'examples', 'models')

# analysis/rdm
rdm_analysisfile = os.path.join(analysispath, 'rdm.py')
rdm_analysis     = imp.load_source('rdm_analysis', rdm_analysisfile)

# models/rdm_fixed
rdm_fixed_modelfile  = os.path.join(modelspath, 'rdm_fixed.py')
rdm_fixed_model      = imp.load_source('rdm_fixed_model', rdm_fixed_modelfile)
rdm_fixed_behavior   = os.path.join(trialspath, 'rdm_fixed', 'trials_behavior.pkl')
rdm_fixed_activity   = os.path.join(trialspath, 'rdm_fixed', 'trials_activity.pkl')

#=========================================================================================

w = utils.mm_to_inch(174)
r = 0.29
h = r*w

fig = Figure(w=w, h=h, labelpadx=4.5, labelpady=4.5)

#=========================================================================================

w_task     = 0.26
w_behavior = 0.205
w_activity = 0.205

h = 0.67
h_epochs = 0.2
h_input  = 0.14

x0 = 0.145
DX = 0.085
dx = 0.075

y0      = 0.22
dy_task = 0.14
dy      = 0.08

fig.add('task',                      [x0, y0, w_task, h_epochs])
fig.add('stimulus',                  [fig['task'].x, fig['task'].top+dy_task, w_task, h_input])
fig.add('fixation',                  [fig['task'].x, fig['stimulus'].top+dy, w_task, h_input])
fig.add('correct-stimulus-duration', [fig['task'].right+DX, fig['task'].y, w_behavior, h], 'none')
fig.add('on-stimulus',               [fig['correct-stimulus-duration'].right+dx, fig['task'].y, w_activity, h])

#=========================================================================================

fontsize = 7.5

plot = fig['fixation']
plot.text(-380, 0.5, 'Fixation cue', ha='right', va='center', fontsize=fontsize)

plot = fig['stimulus']
plot.text(-380, 0.5, 'Evidence L/R', ha='right', va='center', fontsize=fontsize)

#=========================================================================================
# Task
#=========================================================================================

lw = 1.1
ms = 5

durations = {
    'fixation': (0, 600),
    'stimulus': (600, 1400),
    'decision': (1400, 2000)
    }

tmax = durations['decision'][-1]
time = np.linspace(0, tmax, 151)[1:]

#=========================================================================================

plot = fig['fixation']
plot.axis_off('bottom')

fixation = np.zeros_like(time)
for i, t in enumerate(time):
    if t < durations['stimulus'][1]:
        fixation[i] = 1
plot.plot(time, fixation, color=Figure.colors('magenta'), lw=lw)

plot.yticks([0, 1])
plot.yticklabels(['OFF', 'ON'], fontsize=5.5)

plot.xlim(0, tmax)
plot.ylim(0, 1)

plot.text(durations['decision'][0], 1.1, '\"Go\"', ha='left', va='bottom', fontsize=7)

#=========================================================================================

plot = fig['stimulus']
plot.axis_off('bottom')

# Stimulus
coh = 25.6
rng = np.random.RandomState(1)

high = np.zeros_like(time)
low  = np.zeros_like(time)
for i, t in enumerate(time):
    if durations['stimulus'][0] <= t < durations['stimulus'][1]:
        high[i] = rdm_fixed_model.scale(+coh) + rng.normal(scale=0.15)
        low[i]  = rdm_fixed_model.scale(-coh) + rng.normal(scale=0.15)

eps = 0.04
plot.plot(time, high+eps, color=Figure.colors('blue'), lw=lw)
plot.plot(time, low,  color=Figure.colors('red'), lw=lw)

plot.yticks([0, 1])
plot.yticklabels([0, 1], fontsize=7)

plot.xlim(0, tmax)
plot.ylim(0, 1)

#=========================================================================================

def circle(x, y, color):
    ms = 4

    if color is None:
        plot.plot(x, y, 'o', ms=ms, mew=0.5, mfc='none', mec='k')
    else:
        plot.plot(x, y, 'o', ms=ms, mew=0.5, mfc=color, mec=color)

def display_actions(plot, x, y, rewards, colors):
    dx_circ  = 120
    dy_above = +0.14
    dy_below = -0.16
    fontsize = 6

    rF, rL, rR = rewards
    cF, cL, cR = colors

    circle(x, y, cF)
    plot.text(x, y+dy_above, 'F', ha='center', va='bottom', fontsize=fontsize)
    plot.text(x, y+dy_below, rF, ha='center', va='top', fontsize=fontsize)

    circle(x-dx_circ, y, cL)
    plot.text(x-dx_circ, y+dy_above, 'L', ha='center', va='bottom', fontsize=fontsize)
    plot.text(x-dx_circ, y+dy_below, rL, ha='center', va='top', fontsize=fontsize)

    circle(x+dx_circ, y, cR)
    plot.text(x+dx_circ, y+dy_above, 'R', ha='center', va='bottom', fontsize=fontsize)
    plot.text(x+dx_circ, y+dy_below, rR, ha='center', va='top', fontsize=fontsize)

#-----------------------------------------------------------------------------------------
"""
plot = fig['task']
plot.axis_off('left')
plot.axis_off('bottom')

y = 0
plot.plot([0, tmax], y*np.ones(2), color='k', lw=0.75)
for t in [0] + [durations[e][1] for e in ['fixation', 'stimulus', 'decision']]:
    plot.plot(t*np.ones(2), [y-0.05, y+0.05], 'k', lw=0.75)

y_reward = 1.1

# Rewards - fixation
rewards = ['0', '-1', '-1']
colors  = ['k', None, None]
display_actions(plot, np.mean(durations['fixation']), y_reward, rewards, colors)

# Rewards - stimulus
rewards = ['0', '-1', '-1']
colors  = ['k', None, None]
display_actions(plot, np.mean(durations['stimulus']), y_reward, rewards, colors)

# Rewards - decision
rewards = ['0', '+1', '0']
colors  = [None, Figure.colors('darkblue'), None]
display_actions(plot, np.mean(durations['decision']), y_reward, rewards, colors)

# Epoch labels
for e, label in zip(['fixation', 'stimulus', 'decision'],
                    ['Fixation', 'Stimulus', 'Decision']):
    plot.text(np.mean(durations[e]), y+0.16, label, ha='center', va='bottom',
              fontsize=7)

# Epoch durations
for e, label in zip(['fixation', 'stimulus', 'decision'],
                    ['750 ms', '80-1500 ms', '500 ms']):
    plot.text(np.mean(durations[e]), y-0.2, label, ha='center', va='top',
              fontsize=7)

plot.yticks()

plot.xlim(0, tmax)
plot.ylim(0, 1)
"""
#=========================================================================================
"""
plot = fig['correct-stimulus-duration']

kwargs = {'ms': 4, 'lw': 1}
rdm_analysis.correct_stimulus_duration(rdm_fixed_behavior, plot, nbins=10, **kwargs)

plot.xlim(0, 1200)
plot.xticks([0, 400, 800, 1200])

plot.ylim(0.5, 1)
plot.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
plot.yticklabels([0.5, 0.6, 0.7, 0.8, 0.9, '1'])

plot.xlabel('Stimulus duration (ms)')
plot.ylabel('Probability correct')
"""
#=========================================================================================

plot = fig['on-stimulus']

unit = 11

#kwargs = {'on-stimulus-tmin': -200, 'on-stimulus-tmax': 450, 'colors': 'kiani',
#          'dashes': [3.5, 2]}
#rdm_analysis.sort(rdm_fixed_activity, {'on-stimulus': plot}, unit=unit, **kwargs)

kwargs = {'on-stimulus-tmin': -200, 'on-stimulus-tmax': 600,
          'on-choice-tmin': -400, 'on-choice-tmax': 0,
          'colors': 'kiani', 'dashes': [3.5, 2]}
rdm_analysis.sort_return(rdm_fixed_activity, fig.plots, **kwargs)

plot.xlim(-200, 600)
plot.xticks([-200, 0, 200, 400, 600])

#plot.ylim(0, 2, 3)
#plot.yticks([0, 1, 2, 3])
plot.ylim(0.5, 1.2)
plot.yticks([0.5, 0.75, 1])
plot.yticklabels(['0.5', '0.75', '1'])

plot.xlabel('Time from stimulus (ms)')
plot.ylabel('Expected return')

# Legend
#props = {'prop': {'size': 6}, 'handlelength': 1.2,
#         'handletextpad': 1.1, 'labelspacing': 0.7}
#plot.legend(bbox_to_anchor=(0.43, 1.2), **props)

#=========================================================================================

fig.save()
