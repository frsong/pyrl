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

# Analysis
analysisfile = os.path.join(analysispath, 'padoaschioppa2006.py')
analysis     = imp.load_source('padoaschioppa2006', analysisfile)

# Model
modelfile    = os.path.join(modelspath, 'padoaschioppa2006.py')
model        = imp.load_source('model', modelfile)
trialsfile_b = os.path.join(trialspath, 'padoaschioppa2006', 'trials_behavior.pkl')
trialsfile_a = os.path.join(trialspath, 'padoaschioppa2006', 'trials_activity.pkl')

# Model 2
modelfile2    = os.path.join(modelspath, 'padoaschioppa2006_1A3B.py')
model2        = imp.load_source('model2', modelfile2)
trialsfile2_b = os.path.join(trialspath, 'padoaschioppa2006_1A3B', 'trials_behavior.pkl')
trialsfile2_a = os.path.join(trialspath, 'padoaschioppa2006_1A3B', 'trials_activity.pkl')

#=========================================================================================
# Figure
#=========================================================================================

w   = utils.mm_to_inch(174)
r   = 0.52
fig = Figure(w=w, r=r, axislabelsize=8.5, ticklabelsize=6.5, labelpadx=4.5, labelpady=4.5)

x0 = 0.07
y0 = 0.17

w = 0.18
h = 0.3

DX = 0.09
dx = 0.04
DY = 0.16

fig.add('choice-lower', [x0, y0, w, h])
fig.add('choice-upper', [fig[-1].x, fig[-1].top+DY, w, h])

fig.add('activity-1', [fig['choice-upper'].right+DX, fig['choice-upper'].y, w, h])
fig.add('activity-2', [fig[-1].right+dx, fig[-1].y, w, h])
fig.add('activity-3', [fig[-1].right+dx, fig[-1].y, w, h])

fig.add('activity-4', [fig['choice-lower'].right+DX, fig['choice-lower'].y, w, h])
fig.add('activity-5', [fig[-1].right+dx, fig[-1].y, w, h])
fig.add('activity-6', [fig[-1].right+dx, fig[-1].y, w, h])

plotlabels = {
    'A': (0.01, 0.935),
    'B': (0.28,  0.935)
    }
fig.plotlabels(plotlabels)

#=========================================================================================

plot = fig['choice-upper']

kwargs = {'ms': 4.5, 'lw': 1.25}
analysis.choice_pattern(trialsfile_b, model.offers, plot, **kwargs)

plot.yticks([0, 50, 100])

plot.text_upper_left('1A = {}B'.format(model.A_to_B), fontsize=7.5,
                     color=Figure.colors('green'))

#=========================================================================================

plot = fig['choice-lower']

kwargs = {'ms': 4.5, 'lw': 1.25}
analysis.choice_pattern(trialsfile2_b, model2.offers, plot, **kwargs)

plot.yticks([0, 50, 100])

plot.xlabel('Offer (\#B : \#A)')
plot.ylabel('Percent choice B')

plot.text_upper_left('1A = {}B'.format(model2.A_to_B), fontsize=7.5,
                     color=Figure.colors('green'))

#=========================================================================================

kwargs = {'ms': 4.5, 'lw': 1.25}

plots = [fig['activity-1'], fig['activity-2'], fig['activity-3']]
units = [87, 22, 51]
analysis.sort_epoch(trialsfile_b, trialsfile_a, 'prechoice', model.offers, plots, units,
                    network='v', **kwargs)

plots = [fig['activity-4'], fig['activity-5']]
units = [42, 30]
analysis.sort_epoch(trialsfile_b, trialsfile_a, 'prechoice', model.offers, plots, units,
                    network='v', **kwargs)

plots = [fig['activity-6']]
units = [35]
analysis.sort_epoch(trialsfile_b, trialsfile_a, 'prechoice', model.offers, plots, units,
                    network='v', separate_by_choice=True, **kwargs)

plot = fig['activity-4']
plot.xlabel('Offer (\#B : \#A)')
plot.ylabel('Firing rate (a.u.)')

#-----------------------------------------------------------------------------------------
# Labels
#-----------------------------------------------------------------------------------------

fontsize = 8.5
dy       = 0.04
fig['activity-1'].text_upper_center('Chosen value', dy=dy, fontsize=fontsize)
fig['activity-2'].text_upper_center('Chosen value', dy=dy, fontsize=fontsize)
fig['activity-3'].text_upper_center('Chosen value', dy=dy, fontsize=fontsize)
fig['activity-4'].text_upper_center('Offer value',  dy=dy, fontsize=fontsize)
fig['activity-5'].text_upper_center('Offer value',  dy=dy, fontsize=fontsize)
fig['activity-6'].text_upper_center('Choice',       dy=dy, fontsize=fontsize)

#=========================================================================================

fig.save()
