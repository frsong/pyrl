import imp
import os
import sys

import numpy as np

from pyrl          import utils
from pyrl.figtools import Figure, mpl

#=========================================================================================
# Files
#=========================================================================================

here   = utils.get_here(__file__)
parent = utils.get_parent(here)

# Paths
scratchpath  = os.path.join('/Users', 'francis', 'scratch', 'work', 'pyrl')
trialspath   = scratchpath
analysispath = os.path.join(parent, 'examples', 'analysis')
modelspath   = os.path.join(parent, 'examples', 'models')

# Analysis
analysisfile = os.path.join(analysispath, 'padoa_schioppa2006.py')
analysis     = imp.load_source('padoa_schioppa2006_analysis', analysisfile)

# Model
modelfile    = os.path.join(modelspath, 'padoa_schioppa2006.py')
model        = imp.load_source('model', modelfile)
trialsfile_b = os.path.join(trialspath, 'padoa_schioppa2006', 'trials_behavior.pkl')
trialsfile_e = os.path.join(trialspath, 'padoa_schioppa2006', 'trials_electrophysiology.pkl')

#=========================================================================================
# Figure
#=========================================================================================

w = utils.mm_to_inch(174)
r = 0.3
h = r*w
fig = Figure(w=w, h=h, axislabelsize=8, ticklabelsize=6, labelpadx=4.5, labelpady=4)

x0 = 0.07
y0 = 0.27
DX = 0.08

w_choice = 0.2
h_choice = 0.63

w_sorted = 0.17
h_sorted = h_choice
dx       = 0.055

plots = {
    'choice':    fig.add([x0, y0, w_choice, h_choice]),
    'sorted-cv': fig.add([x0+w_choice+DX, y0, w_sorted, h_sorted]),
    'sorted-ov': fig.add([x0+w_choice+DX+w_sorted+dx, y0, w_sorted, h_sorted]),
    'sorted-cj': fig.add([x0+w_choice+DX+w_sorted+dx+w_sorted+dx, y0, w_sorted, h_sorted])
    }

#=========================================================================================

plot = plots['choice']

kwargs = {'ms': 4.5, 'lw': 1.25, 'rotation': 60}
analysis.choice_pattern(trialsfile_b, model, plot, **kwargs)

plot.xlabel('Offer (\#B : \#A)')
plot.ylabel('Percent choice B')

plot.text_upper_left('1A = {}B'.format(model.A_to_B), fontsize=7)

#=========================================================================================

dy       = 0.03
fontsize = 7

#=========================================================================================

plot = plots['sorted-cv']

unit = 33
kwargs = {'ms': 4.5, 'lw': 1.25, 'rotation': 60}
analysis.sort_epoch('postoffer', trialsfile_e, model, plot, unit=unit,
                    network='baseline', **kwargs)

plot.xlabel('Offer (\#B : \#A)')
plot.ylabel('Firing rate (a.u.)', labelpad=6.5)

plot.text_upper_center('Chosen value', dy=dy, fontsize=fontsize)

#=========================================================================================

plot = plots['sorted-ov']

unit = 4
kwargs = {'ms': 4.5, 'lw': 1.25, 'rotation': 60}
analysis.sort_epoch('postoffer', trialsfile_e, model, plot, unit=unit,
                    network='baseline', **kwargs)

plot.text_upper_center('Offer value', dy=dy, fontsize=fontsize)

#=========================================================================================

plot = plots['sorted-cj']

unit = 22
kwargs = {'ms': 4.5, 'lw': 1.25, 'rotation': 60, 'min_trials': 50}
analysis.sort_epoch_choice('late-delay', trialsfile_e, model, plot, unit=unit,
                           network='policy', **kwargs)

plot.text_upper_center('Chosen juice', dy=dy, fontsize=fontsize)

#=========================================================================================

fig.save()
