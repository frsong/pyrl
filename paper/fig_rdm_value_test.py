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
r = 0.5
fig = Figure(w=w, r=r)

x0 = 0.12
y0 = 0.15
w  = 0.37
h  = 0.8
dx = 1.3*w

fig.add('on-stimulus', [x0,    y0, w, h])
fig.add('on-choice',   [x0+dx, y0, w, h])

#=========================================================================================

kwargs = {'on-stimulus-tmin': -200, 'on-stimulus-tmax': 600,
          'on-choice-tmin': -400, 'on-choice-tmax': 0,
          'colors': 'kiani', 'dashes': [3.5, 2]}
rdm_analysis.sort_return(rdm_fixed_activity, fig.plots, **kwargs)

plot = fig['on-stimulus']
plot.xlim(-200, 600)
plot.xticks([-200, 0, 200, 400, 600])
plot.ylim(0.5, 1.1)
#plot.yticks([0.5, 1])

plot.xlabel('Time from stimulus (ms)')
plot.ylabel('Expected reward')

# Legend
props = {'prop': {'size': 8}, 'handlelength': 1.2,
         'handletextpad': 1.1, 'labelspacing': 0.7}
plot.legend(bbox_to_anchor=(0.33, 1), **props)

plot = fig['on-choice']
plot.xlim(-400, 0)
plot.xticks([-400, -200, 0])
plot.ylim(0.5, 1.1)
#plot.yticks([0.5, 1])

plot.xlabel('Time from decision (ms)')

#=========================================================================================

fig.save()
