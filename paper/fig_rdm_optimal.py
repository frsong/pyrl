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
rdm_analysisfile = os.path.join(analysispath, 'rdm.py')
rdm_analysis     = imp.load_source('rdm_analysis', rdm_analysisfile)

# models/rdm_fixed
rdm_fixed_modelfile  = os.path.join(modelspath, 'rdm_fixed.py')
rdm_fixed_model      = imp.load_source('rdm_fixed_model', rdm_fixed_modelfile)
rdm_fixed_behavior   = os.path.join(trialspath, 'rdm_fixed', 'trials_behavior.pkl')
rdm_fixed_activity   = os.path.join(trialspath, 'rdm_fixed', 'trials_activity.pkl')

#=========================================================================================

w   = utils.mm_to_inch(174)
r   = 1
fig = Figure(w=w, r=r)

x0 = 0.1
y0 = 0.1

dy = 0.05

w = 0.82
h = 0.13

fig.add('trial-5', [x0, y0, w, h])
fig.add('trial-4', [x0, fig[-1].top+dy, w, h])
fig.add('trial-3', [x0, fig[-1].top+dy, w, h])
fig.add('trial-2', [x0, fig[-1].top+dy, w, h])
fig.add('trial-1', [x0, fig[-1].top+dy, w, h])

#=========================================================================================

trials, U, Z, A, rho, M, perf, r_policy, r_value = utils.load(rdm_fixed_activity)

inputs = rdm_fixed_model.inputs

def process_trial(plot, n):
    if perf.choices[n] is None:
        print("Trial {}: No decision.".format(n))
        return

    trial = trials[n]
    time  = trial['time']
    u = U[:,n]
    z = Z[:,n]

    stimulus  = np.asarray(trial['epochs']['stimulus'])
    evidenceL = np.sum(u[stimulus-1,inputs['LEFT']])
    evidenceR = np.sum(u[stimulus-1,inputs['RIGHT']])

    decision = np.asarray(trial['epochs']['decision'])
    t_choice = perf.t_choices[n]
    idx      = decision[np.where(decision <= t_choice)]
    t0       = time[idx][0]

    pL = z[idx,inputs['LEFT']]
    pR = z[idx,inputs['RIGHT']]
    S  = pL + pR

    if perf.choices[n] == 'R':
        ls = '-'
    #else:
    #    ls = '--'
        plot.plot(time[idx]-t0, pL/S, ls, color=Figure.colors('red'), lw=0.5, zorder=5)

    if perf.choices[n] == 'R':
        ls = '-'
    #else:
    #    ls = '--'
        plot.plot(time[idx]-t0, pR/S, ls, color=Figure.colors('blue'), lw=0.5, zorder=5)

    #pL_opt = np.exp(evidenceL)
    #pR_opt = np.exp(evidenceR)
    #S_opt  = pL_opt + pR_opt

    #plot.plot(time[idx]-t0, pL_opt/S_opt*np.ones(len(idx)), '--', color=Figure.colors('red'), lw=1.5)
    #plot.plot(time[idx]-t0, pR_opt/S_opt*np.ones(len(idx)), '--', color=Figure.colors('blue'), lw=1.5)

    #stimulus_duration = np.ptp(trial['durations']['stimulus'])
    #plot.plot(stimulus_duration, pL_opt/pR_opt, 'o', color='k')

    #plot.ylim(0, 1)

M = 0
for n, trial in enumerate(trials):
    if trial['left_right'] > 0 and trial['coh'] == 0:
        process_trial(fig['trial-1'], n)
        M += 1
    if trial['left_right'] > 0 and trial['coh'] == 6.4:
        process_trial(fig['trial-2'], n)
        M += 1
    if trial['left_right'] > 0 and trial['coh'] == 12.8:
        process_trial(fig['trial-3'], n)
        M += 1
    if trial['left_right'] > 0 and trial['coh'] == 25.6:
        process_trial(fig['trial-4'], n)
        M += 1
    if trial['left_right'] > 0 and trial['coh'] == 51.2:
        process_trial(fig['trial-5'], n)
        M += 1

    if M == 5*100:
        print("Too many!")
        break

#=========================================================================================

fig['trial-5'].xlabel('Time (ms)')
fig['trial-5'].ylabel('$P(a)$')

#=========================================================================================
'''
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
fig.add('correct-stimulus-duration', [fig['task'].right+DX, fig['task'].y, w_behavior, h])
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
                    ['250-750 ms', '80-1500 ms', '500 ms']):
    plot.text(np.mean(durations[e]), y-0.2, label, ha='center', va='top',
              fontsize=7)

plot.yticks()

plot.xlim(0, tmax)
plot.ylim(0, 1)

#=========================================================================================

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

#=========================================================================================

plot = fig['on-stimulus']

unit = 11

kwargs = {'on-stimulus-tmin': -200, 'on-stimulus-tmax': 400, 'colors': 'kiani',
          'dashes': [3.5, 2]}
rdm_analysis.sort(rdm_fixed_activity, {'on-stimulus': plot}, unit=unit, **kwargs)

plot.xlim(-200, 400)
plot.xticks([-200, 0, 200, 400])

plot.yticks([0, 1, 2])

plot.xlabel('Time from stimulus (ms)')
plot.ylabel('Firing rate (a.u.)')

# Legend
props = {'prop': {'size': 6}, 'handlelength': 1.2,
         'handletextpad': 1.1, 'labelspacing': 0.7}
plot.legend(bbox_to_anchor=(0.41, 1.2), **props)
'''
#=========================================================================================

fig.save()
