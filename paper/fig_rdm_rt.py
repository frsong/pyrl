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
trialspath   = os.path.join(scratchpath, 'work', 'pyrl', 'examples')
analysispath = os.path.join(parent, 'examples', 'analysis')
modelspath   = os.path.join(parent, 'examples', 'models')

# analysis/rdm
analysisfile = os.path.join(analysispath, 'rdm.py')
analysis     = imp.load_source('analysis', analysisfile)

# models/rdm_rt
modelfile  = os.path.join(modelspath, 'rdm_rt.py')
model      = imp.load_source('model', modelfile)
behavior   = os.path.join(trialspath, 'rdm_rt', 'trials_behavior.pkl')
activity   = os.path.join(trialspath, 'rdm_rt', 'trials_activity.pkl')

#=========================================================================================

w = utils.mm_to_inch(174)
r = 0.29
h = r*w

fig = Figure(w=w, h=h, labelpadx=4.5, labelpady=4.5)

#=========================================================================================

w_task     = 0.26
w_behavior = 0.19
w_activity = 0.19

h = 0.67
h_epochs = 0.2
h_input  = 0.14

x0 = 0.145
DX = 0.095
dx = 0.075

y0      = 0.22
dy_task = 0.14
dy      = 0.08

fig.add('task',         [x0, y0, w_task, h_epochs])
fig.add('stimulus',     [fig['task'].x, fig['task'].top+dy_task, w_task, h_input])
fig.add('fixation',     [fig['task'].x, fig['stimulus'].top+dy, w_task, h_input])
fig.add('chronometric', [fig['task'].right+DX, fig['task'].y, w_behavior, h])
fig.add('on-stimulus',  [fig[-1].right+dx, fig['task'].y, w_activity, h])

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
    'stimulus': (600, 2000)
    }

tmax = durations['stimulus'][-1]
time = np.linspace(0, tmax, 151)[1:]

#=========================================================================================

plot = fig['fixation']
plot.axis_off('bottom')

fixation = np.zeros_like(time)
for i, t in enumerate(time):
    if t < durations['stimulus'][0]:
        fixation[i] = 1
plot.plot(time, fixation, color=Figure.colors('magenta'), lw=lw)

plot.yticks([0, 1])
plot.yticklabels(['OFF', 'ON'], fontsize=5.5)

plot.xlim(0, tmax)
plot.ylim(0, 1)

plot.text(durations['stimulus'][0], 1.1, '\"Go\"', ha='left', va='bottom', fontsize=7)

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
        high[i] = model.scale(+coh) + rng.normal(scale=0.15)
        low[i]  = model.scale(-coh) + rng.normal(scale=0.15)

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

plot = fig['task']
plot.axis_off('left')
plot.axis_off('bottom')

y = 0
plot.plot([0, tmax], y*np.ones(2), color='k', lw=0.75)
for t in [0] + [durations[e][1] for e in ['fixation', 'stimulus']]:
    plot.plot(t*np.ones(2), [y-0.05, y+0.05], 'k', lw=0.75)

y_reward = 1.1

# Rewards - fixation
rewards = ['0', '-1', '-1']
colors  = ['k', None, None]
display_actions(plot, np.mean(durations['fixation']), y_reward, rewards, colors)

# Rewards - stimulus
rewards = ['0', '+1', '0']
colors  = [None, Figure.colors('darkblue'), None]
display_actions(plot, np.mean(durations['stimulus']), y_reward, rewards, colors)

# Rewards - decision
#rewards = ['0', '+1', '0']
#colors  = [None, Figure.colors('darkblue'), None]
#display_actions(plot, np.mean(durations['decision']), y_reward, rewards, colors)

# Epoch labels
for e, label in zip(['fixation', 'stimulus'],
                    ['Fixation', 'Stimulus/decision']):
    plot.text(np.mean(durations[e]), y+0.16, label, ha='center', va='bottom',
              fontsize=7)

# Epoch durations
for e, label in zip(['fixation'],
                    ['750 ms']):
    plot.text(np.mean(durations[e]), y-0.2, label, ha='center', va='top',
              fontsize=7)

plot.yticks()

plot.xlim(0, tmax)
plot.ylim(0, 1)

#=========================================================================================

plot = fig['chronometric']

kwargs = {'ms': 4, 'lw': 1}
analysis.chronometric(behavior, plot, **kwargs)

plot.ylim(200, 1000)
plot.yticks([200, 400, 600, 800, 1000])

plot.xlabel('Percent coherence')
plot.ylabel('Reaction time (ms)', labelpad=3)

#=========================================================================================

plot = fig['on-stimulus']

unit = 2#61

kwargs = {'on-stimulus-tmin': -200, 'on-stimulus-tmax': 400, 'colors': 'kiani',
          'dashes': [3.5, 2]}
analysis.sort(activity, {'on-stimulus': plot}, unit=unit, **kwargs)

plot.xlim(-200, 400)
plot.xticks([-200, 0, 200, 400])

plot.ylim(0, 2)
plot.yticks([0, 1, 2])

plot.xlabel('Time from stimulus (ms)')
plot.ylabel('Firing rate (a.u.)')

# Legend
props = {'prop': {'size': 6}, 'handlelength': 1.2,
         'handletextpad': 1.05, 'labelspacing': 0.7}
plot.legend(bbox_to_anchor=(0.44, 1.2), **props)

#=========================================================================================

fig.save()
