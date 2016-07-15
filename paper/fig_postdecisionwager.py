from __future__ import division

import imp
import os
import sys

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

# analysis
analysisfile = os.path.join(analysispath, 'postdecisionwager.py')
analysis     = imp.load_source('postdecisionwager_analysis', analysisfile)

# model
modelfile    = os.path.join(modelspath, 'postdecisionwager.py')
model        = imp.load_source('model', modelfile)
trialsfile_b = os.path.join(trialspath, 'postdecisionwager', 'trials_behavior.pkl')
trialsfile_a = os.path.join(trialspath, 'postdecisionwager', 'trials_activity.pkl')

#=========================================================================================
# Figure
#=========================================================================================

w = utils.mm_to_inch(114)
r = 0.9
thickness     = 0.4
axislabelsize = 6
fig = Figure(w=w, r=r, thickness=thickness, ticksize=3, ticklabelsize=5,
             axislabelsize=axislabelsize, labelpadx=3, labelpady=4.5)

x0 = 0.16
y0 = 0.67
DX = 0.1

w_task = 0.82
h_task = 0.3

dy0 = 0.09
dy  = 0.12

w_behavior = 0.24
h_behavior = 0.2
y1 = y0-dy0-h_behavior

dx   = 0.03
w_fr = 0.12
h_fr = 0.18
y2 = y1-dy-h_fr

fig.add('task',                      [x0, y0, w_task, h_task]),
fig.add('sure-stimulus-duration',    [x0, y1, w_behavior, h_behavior]),
fig.add('correct-stimulus-duration', [fig[-1].right+DX, y1, w_behavior, h_behavior]),
fig.add('noTs-stimulus',             [x0, y2, w_fr, h_fr]),
fig.add('noTs-choice',               [fig[-1].right+dx, y2, 5/8*w_fr, h_fr]),
fig.add('Ts-stimulus',               [fig[-1].right+1.1*DX, y2, w_fr, h_fr]),
fig.add('Ts-sure',                   [fig[-1].right+dx, y2, w_fr, h_fr]),
fig.add('Ts-choice',                 [fig[-1].right+dx, y2, 5/8*w_fr, h_fr])

pl_x0 = 0.025
pl_y0 = 0.945
pl_y1 = 0.595
pl_y2 = 0.28
plotlabels = {
    'A': (pl_x0, pl_y0),
    'B': (pl_x0, pl_y1),
    'C': (pl_x0, pl_y2)
    }
fig.plotlabels(plotlabels, fontsize=9)

#=========================================================================================
# Task
#=========================================================================================

rng = np.random.RandomState(1)

plot = fig['task']
plot.axis_off('left')
plot.axis_off('bottom')

ms       = 2.5
dx_circ  = 0.14
dy_above = +0.08
dy_below = -0.1
fontsize = 4.5

def circle(x, y, color):
    if color is None:
        plot.plot(x, y, 'o', ms=ms, mew=0.5, mfc='none', mec='k')
    else:
        plot.plot(x, y, 'o', ms=ms, mew=0.5, mfc=color, mec=color)

def display_actions(x, y, rewards, colors):
    rF, rL, rR, rS = rewards
    cF, cL, cR, cS = colors

    circle(x-1.5*dx_circ, y, cF)
    plot.text(x-1.5*dx_circ, y+dy_above, 'F', ha='center', va='bottom', fontsize=fontsize)
    plot.text(x-1.5*dx_circ, y+dy_below, rF, ha='center', va='top', fontsize=fontsize)

    circle(x-0.5*dx_circ, y, cL)
    plot.text(x-0.5*dx_circ, y+dy_above, 'L', ha='center', va='bottom', fontsize=fontsize)
    plot.text(x-0.5*dx_circ, y+dy_below, rL, ha='center', va='top', fontsize=fontsize)

    circle(x+0.5*dx_circ, y, cR)
    plot.text(x+0.5*dx_circ, y+dy_above, 'R', ha='center', va='bottom', fontsize=fontsize)
    plot.text(x+0.5*dx_circ, y+dy_below, rR, ha='center', va='top', fontsize=fontsize)

    circle(x+1.5*dx_circ, y, cS)
    plot.text(x+1.5*dx_circ, y+dy_above, 'S', ha='center', va='bottom', fontsize=fontsize)
    plot.text(x+1.5*dx_circ, y+dy_below, rS, ha='center', va='top', fontsize=fontsize)

# Durations
durations = {}
durations['fixation'] = (0, 1)
durations['stimulus'] = (1, 2)
durations['delay']    = (2, 4)
durations['sure']     = (3, 4)
durations['decision'] = (4, 5)

trial_t = np.linspace(0, durations['decision'][1], 501)[1:]
lw = 0.6

# Baselines
y_sure_Ts   = 1.75
y_fixation  = 1.3
y_stimulus  = 0.65
y_sure_noTs = 0.05

def fake_left_axis(ybot, ytop, thickness, ticklabels=None, axislabel=None):
    plot.plot(np.zeros(2), [ybot, ytop], color='k', lw=thickness)
    plot.plot([-0.03, 0], ybot*np.ones(2), color='k', lw=thickness)
    plot.plot([-0.03, 0], ytop*np.ones(2), color='k', lw=thickness)

    if ticklabels is not None:
        plot.text(-0.06, ytop, ticklabels[1], ha='right', va='center', fontsize=3.5)
        plot.text(-0.06, ybot, ticklabels[0], ha='right', va='center', fontsize=3.5)

    if axislabel is not None:
        text = plot.text(-0.25, (ybot+ytop)/2, axislabel,
                         ha='right', va='center', fontsize=4.5)

#-----------------------------------------------------------------------------------------
# Plot fixation
#-----------------------------------------------------------------------------------------

fixation = np.zeros_like(trial_t)
w, = np.where((0 <= trial_t) & (trial_t <= durations['decision'][0]))
fixation[w] = 1

def rescale(y):
    return y_fixation + 0.2*y

fake_left_axis(rescale(0), rescale(1), thickness, ticklabels=['OFF', 'ON'],
               axislabel='Fixation cue')
plot.plot(trial_t, rescale(fixation), color=Figure.colors('darkgreen'), lw=lw)

#-----------------------------------------------------------------------------------------
# Plot stimulus
#-----------------------------------------------------------------------------------------

coh = 25.6

stimulus_L = np.zeros_like(trial_t)
stimulus_R = np.zeros_like(trial_t)

w, = np.where((durations['stimulus'][0] < trial_t) & (trial_t <= durations['stimulus'][1]))
stimulus_L[w] = model.scale(-coh) + rng.normal(scale=0.15, size=len(w))
stimulus_R[w] = model.scale(+coh) + rng.normal(scale=0.15, size=len(w))

def rescale(y):
    return y_stimulus + 0.3*y

fake_left_axis(rescale(0), rescale(1), thickness, ticklabels=[0, 1],
               axislabel='Evidence L/R')
plot.plot(trial_t, rescale(stimulus_L), color=Figure.colors('red'), lw=lw)
plot.plot(trial_t, rescale(stimulus_R), color=Figure.colors('blue'), lw=lw)

#-----------------------------------------------------------------------------------------
# Plot sure target
#-----------------------------------------------------------------------------------------

sure = np.zeros_like(trial_t)
w, = np.where((durations['sure'][0] < trial_t))
sure[w] = 1

def rescale(y):
    return y_sure_Ts + 0.2*y

fake_left_axis(rescale(0), rescale(1), thickness, ticklabels=['OFF', 'ON'],
               axislabel='Sure target\n(Wager trials)')
plot.plot(trial_t, rescale(sure), color=Figure.colors('magenta'), lw=lw)

def rescale(y):
    return y_sure_noTs + 0.2*y

sure = np.zeros_like(trial_t)
fake_left_axis(rescale(0), rescale(1), thickness, ticklabels=['OFF', 'ON'],
               axislabel='Sure target\n(Non-wager trials)')
plot.plot(trial_t, rescale(sure), color=Figure.colors('magenta'), lw=lw)

#-----------------------------------------------------------------------------------------
# Display actions (all trials))
#-----------------------------------------------------------------------------------------

shift = 0.55

rewards = ['0', '-1', '-1', '-1']
colors  = ['k', None, None, None]
display_actions(np.mean(durations['fixation']), y_stimulus+shift, rewards, colors)

rewards = ['0', '-1', '-1', '-1']
colors  = ['k', None, None, None]
display_actions(np.mean(durations['stimulus']), y_stimulus+shift, rewards, colors)

rewards = ['0', '-1', '-1', '-1']
colors  = ['k', None, None, None]
display_actions(np.mean(durations['delay']), y_stimulus+shift, rewards, colors)

#-----------------------------------------------------------------------------------------
# Display actions (Ts)
#-----------------------------------------------------------------------------------------

shift = 0.5

rewards = ['0', '0', '1', '0.7']
colors  = [None, None, Figure.colors('darkblue'), Figure.colors('blue')]
display_actions(np.mean(durations['decision']), y_sure_Ts+shift, rewards, colors)

plot.text(durations['sure'][0], y_sure_Ts+shift, 'Sure bet offered',
          ha='left', va='center', fontsize=fontsize+0.5)

#-----------------------------------------------------------------------------------------
# Display actions (no Ts)
#-----------------------------------------------------------------------------------------

shift = 0.3

rewards = ['0', '0', '1', '-1']
colors  = [None, None, Figure.colors('darkblue'), None]
display_actions(np.mean(durations['decision']), y_sure_noTs+shift, rewards, colors)

plot.text(durations['sure'][0], y_sure_noTs+shift, 'Sure bet not offered',
          ha='left', va='center', fontsize=fontsize+0.5)

#-----------------------------------------------------------------------------------------

y_timeline = -0.35

# Task timeline
plot.plot([0, durations['decision'][1]], y_timeline*np.ones(2), 'k', lw=0.75)
for t in [0] + [durations[e][1] for e in ['fixation', 'stimulus', 'delay', 'decision']]:
    plot.plot(t*np.ones(2), [y_timeline-0.04, y_timeline+0.04], 'k', lw=0.75)

# Epoch labels
for e, label in zip(['fixation', 'stimulus', 'delay', 'decision'],
                    ['Fixation', 'Stimulus', 'Delay/Sure target', 'Decision']):
    plot.text(np.mean(durations[e]), y_timeline+0.08, label, ha='center', va='bottom',
              fontsize=fontsize+0.5)

# Epoch durations
for e, label in zip(['fixation', 'stimulus', 'delay', 'decision'],
                    ['750 ms', '100-900 ms',
                     '1200-1800 ms\n(Sure target onset 500-750 ms)', '500 ms']):
    plot.text(np.mean(durations[e]), y_timeline-0.11, label, ha='center', va='top',
              fontsize=fontsize+0.5)

# Limits
plot.xlim(0, durations['decision'][1])
plot.ylim(y_timeline, y_sure_Ts+0.2+0.35)

#=========================================================================================

plot = fig['sure-stimulus-duration']

savefile = os.path.join(here, 'work', 'data', 'sure_stimulus_duration.pkl')

if 'fast' in sys.argv and os.path.isfile(savefile):
    print("Plotting data in {}".format(savefile))
    saved = utils.load(savefile)
else:
    saved = None

kwargs = dict(ms=3, lw=0.7)
saved  = analysis.sure_stimulus_duration(trialsfile_b, plot, saved=saved,
                                         nbins=10, **kwargs)
utils.save(savefile, saved)

plot.xticks([100, 300, 500, 700])
plot.yticks([0, 0.2, 0.4, 0.6, 0.8])

plot.xlim(100, 700)
plot.ylim(0, 0.8)

plot.xlabel('Stimulus duration (ms)')
plot.ylabel('Probability sure target')

# Legend
props = {'prop': {'size': 5}, 'handlelength': 1.2,
         'handletextpad': 1.1, 'labelspacing': 0.7}
plot.legend(bbox_to_anchor=(3, 0.98), **props)

#=========================================================================================

plot = fig['correct-stimulus-duration']

savefile = os.path.join(here, 'work', 'data', 'correct_stimulus_duration.pkl')

if 'fast' in sys.argv and os.path.isfile(savefile):
    print("Plotting data in {}".format(savefile))
    saved = utils.load(savefile)
else:
    saved = None

kwargs = dict(ms=2.5, mew=0.5, lw=0.7, dashes=[3, 1])
saved  = analysis.correct_stimulus_duration(trialsfile_b, plot,
                                            saved=saved, nbins=10, **kwargs)
utils.save(savefile, saved)

plot.xticks([100, 300, 500, 700])

plot.xlim(100, 700)
plot.ylim(0.5, 1)

plot.xlabel('Stimulus duration (ms)')
plot.ylabel('Probability correct')

#=========================================================================================
# Plot unit
#=========================================================================================

unit = 63

plots = {name: fig[name]
         for name in ['noTs-stimulus', 'noTs-choice',
                      'Ts-stimulus', 'Ts-sure', 'Ts-choice']}
kwargs = dict(lw=0.8, lw_vline=0.4, dashes=[3, 1.5], dashes_vline=[5, 4])
y = analysis.sort(trialsfile_a, plots, unit=unit, **kwargs)

for plot in plots.values():
    ylim = plot.lim('y', y, lower=0)
    plot.vline(0, lw=thickness, linestyle='--', dashes=[3.5, 2])

fig['noTs-choice'].xticks([-400, 0])
fig['Ts-choice'].xticks([-400, 0])

#=========================================================================================

fontsize_epoch = 5
y_epoch        = 1.03*ylim[1]

plot = fig['noTs-stimulus']
plot.xlabel('Time (ms)')
plot.ylabel('Firing rate (a.u.)')
plot.text(0, y_epoch , 'stimulus', fontsize=fontsize_epoch, ha='left', va='bottom')

plot = fig['noTs-choice']
plot.axis_off('left')
plot.text(0, y_epoch , 'choice', fontsize=fontsize_epoch, ha='right', va='bottom')

plot = fig['Ts-stimulus']
plot.xlabel('Time (ms)')
plot.ylabel('Firing rate (a.u.)')
plot.text(0, y_epoch , 'stimulus', fontsize=fontsize_epoch, ha='left', va='bottom')

plot = fig['Ts-sure']
plot.axis_off('left')
plot.text(0, y_epoch , 'sure target', fontsize=fontsize_epoch, ha='left', va='bottom')

plot = fig['Ts-choice']
plot.axis_off('left')
plot.text(0, y_epoch , 'choice', fontsize=fontsize_epoch, ha='right', va='bottom')

#=========================================================================================

fig.save()
