import numpy as np

from pycog.figtools import Figure

w = 174/25.4
r = 0.6
h = r*w
fig = Figure(w=w, h=h)

w  = 0.4
h  = 0.18
x0 = 0.07
y0 = 0.72
dx = 1.23*w
dy = 1.25*h
plots = {
    'SL-f': fig.add([x0,    y0,      w, h]),
    'SL-s': fig.add([x0,    y0-dy,   w, h]),
    'SL-e': fig.add([x0,    y0-2*dy, w, h], 'none'),
    'SL-o': fig.add([x0,    y0-3*dy, w, h]),
    'RL-f': fig.add([x0+dx, y0,      w, h]),
    'RL-s': fig.add([x0+dx, y0-dy,   w, h]),
    'RL-e': fig.add([x0+dx, y0-2*dy, w, h], 'none'),
    'RL-o': fig.add([x0+dx, y0-3*dy, w, h], 'none')
    }

x0 = 0.01
x1 = x0 + dx
y0 = 0.95
plotlabels = {
    'A': (x0, y0),
    'B': (x1, y0)
    }
fig.plotlabels(plotlabels, fontsize=12.5)

offset = 0.02

#-----------------------------------------------------------------------------------------
# Task structure
#-----------------------------------------------------------------------------------------

fixation  = (0,   250)
stimulus = (250, 750)
decision = (750, 1000)
tmax     = decision[-1]

names  = ['Fixation', 'Stimulus', 'Decision']
epochs = [fixation, stimulus, decision]

dt = 0.1
n_display = int(5/dt)

def plot_epochs(plot):
    plot.plot([0, tmax], np.zeros(2), 'k', lw=1)
    for x in [fixation[0], stimulus[0], decision[0], tmax]:
        plot.plot(x*np.ones(2), [-0.05, 0.05], 'k', lw=1)

    for name, epoch in zip(names, epochs):
        plot.text(np.mean(epoch), 0.1, name, ha='center', va='bottom', fontsize=8)
    plot.text(np.mean(stimulus), 0.3, '(Maintain fixation)',
              ha='center', va='bottom', fontsize=8)

    plot.xlim(0, tmax)
    plot.ylim(-0.5, 0.5)

def plot_fixation(plot):
    plot.axis_off('bottom')

    x  = np.linspace(dt, tmax, int(tmax/dt))
    y  = np.zeros_like(x)
    for i in xrange(len(x)):
        if fixation[0] < x[i] <= fixation[1] or stimulus[0] < x[i] <= stimulus[1]:
            y[i] = 1
    plot.plot(x, y, 'k', lw=1.5)

    plot.xlim(0, tmax)
    plot.ylim(0, 1)
    plot.yticks([0, 1])

def plot_stimulus(plot, rng, coh=+12.8):
    plot.axis_off('bottom')

    x = np.linspace(dt, tmax, int(tmax/dt))
    y_high = np.zeros_like(x)
    y_low  = np.zeros_like(x)
    for i in xrange(len(x)):
        if stimulus[0] < x[i] <= stimulus[1]:
            y_high[i] = (1 + coh/100)/2 + rng.normal(scale=0.1)
            y_low[i]  = (1 - coh/100)/2 + rng.normal(scale=0.1)
    y_high += offset
    plot.plot(x[::n_display], y_high[::n_display], color=Figure.colors('blue'), lw=1.5, zorder=10)
    plot.plot(x[::n_display], y_low[::n_display],  color=Figure.colors('red'),  lw=1.5, zorder=9)

    plot.xlim(0, tmax)
    plot.ylim(0, 1)
    plot.yticks([0, 1])

#-----------------------------------------------------------------------------------------
# Random number generator
#-----------------------------------------------------------------------------------------

rng = np.random.RandomState(1)

#-----------------------------------------------------------------------------------------
# Labels
#-----------------------------------------------------------------------------------------

plots['SL-f'].text_upper_center('Supervised Learning (SL)',    dy=0.25, fontsize=10)
plots['RL-f'].text_upper_center('Reinforcement Learning (RL)', dy=0.25, fontsize=10)

#-----------------------------------------------------------------------------------------
# Supervised
#-----------------------------------------------------------------------------------------

plot = plots['SL-f']
plot_fixation(plot)
plot.ylabel('Fixation cue')

plot = plots['SL-s']
plot_stimulus(plot, rng)
plot.ylabel('Stimulus')

plot = plots['SL-e']
plot_epochs(plot)

plot = plots['SL-o']
plot.axis_off('bottom')

x  = np.linspace(0, tmax, int(tmax/dt)+1)
y_high = np.zeros_like(x)
y_low  = np.zeros_like(x)
for i in xrange(len(x)):
    if decision[0] < x[i] <= decision[1]:
        y_high[i] = 1
y_high += offset
plot.plot(x, y_high, color=Figure.colors('blue'), lw=2, zorder=10, label='Left')
plot.plot(x, y_low,  color=Figure.colors('red'),  lw=2, zorder=9,  label='Right')

plot.xlim(0, tmax)
plot.ylim(0, 1)
plot.yticks([0, 1])
plot.ylabel('Target outputs')

# Legend
props = {'prop': {'size': 7}, 'handlelength': 1.2,
         'handletextpad': 1.1, 'labelspacing': 0.7}
plot.legend(bbox_to_anchor=(0.24, 1), **props)

fontsize = 8
plot.text(np.mean(decision), 0.5, '$z_L > z_R$',
          ha='center', va='center', fontsize=fontsize+1)

#-----------------------------------------------------------------------------------------
# Reinforcement learning
#-----------------------------------------------------------------------------------------

plot = plots['RL-f']
plot_fixation(plot)

plot = plots['RL-s']
plot_stimulus(plot, rng)

plot = plots['RL-e']
plot_epochs(plot)

plot = plots['RL-o']

fontsize = 8
D = 80
Y = 0.68
for epoch in [fixation, stimulus, decision]:
    C = np.mean(epoch)

    # Left
    if epoch == fixation or epoch == stimulus:
        plot.plot(C-D, Y, 'o', mfc='none', mec='k', ms=4.75, mew=0.75)
    elif epoch == decision:
        plot.plot(C-D, Y, 'o', mfc=Figure.colors('blue'), ms=5, mew=0)
    else:
        plot.plot(C-D, Y, 'o', mfc='k', ms=5.5, mew=0)

    # Fixation
    if epoch == fixation or epoch == stimulus:
        plot.plot(C, Y, 'o', mfc='k', ms=5.5, mew=0)
    else:
        plot.plot(C, Y, 'o', mfc='none', mec='k', ms=4.75, mew=0.75)

    # Right
    plot.plot(C+D, Y, 'o', mfc='none', mec='k', ms=4.75, mew=0.75)

    # Labels
    if epoch == decision:
        plot.text(C-D, Y+0.1, 'L', ha='center', va='bottom', fontsize=fontsize,
                  color=Figure.colors('blue'))
    else:
        plot.text(C-D, Y+0.1, 'L', ha='center', va='bottom', fontsize=fontsize)
    plot.text(C,   Y+0.1, 'F', ha='center', va='bottom', fontsize=fontsize)
    plot.text(C+D, Y+0.1, 'R', ha='center', va='bottom', fontsize=fontsize)

# Fixation rewards
C = np.mean(fixation)
plot.text(C-D, Y-0.15, '-1', ha='center', va='top', fontsize=fontsize)
plot.text(C,   Y-0.15, '0',  ha='center', va='top', fontsize=fontsize)
plot.text(C+D, Y-0.15, '-1', ha='center', va='top', fontsize=fontsize)

# Stimulus rewards
C = np.mean(stimulus)
plot.text(C-D, Y-0.15, '-1', ha='center', va='top', fontsize=fontsize)
plot.text(C,   Y-0.15, '0',  ha='center', va='top', fontsize=fontsize)
plot.text(C+D, Y-0.15, '-1', ha='center', va='top', fontsize=fontsize)

# Decision rewards
C = np.mean(decision)
plot.text(C-D, Y-0.15, '+1', ha='center', va='top', fontsize=fontsize, color=Figure.colors('blue'))
plot.text(C,   Y-0.15, '0',  ha='center', va='top', fontsize=fontsize)
plot.text(C+D, Y-0.15, '0',  ha='center', va='top', fontsize=fontsize)

plot.xlim(0, tmax)
plot.ylim(0, 1)

plot.text(-10, Y+0.1,  'Action $a_t$', ha='right', va='bottom', fontsize=fontsize)
plot.text(-10, Y-0.15-0.015, 'Reward $r_t$', ha='right', va='top', fontsize=fontsize)

plot.text(tmax/2, 1.25, 'Maximize reward',
          ha='center', va='center', fontsize=fontsize+1)
plot.text(tmax/2, 0.1, r'Trial ends when $a_t\neq \text{F}$',
          ha='center', va='center', fontsize=fontsize+1)

#-----------------------------------------------------------------------------------------

fig.save()
