from __future__ import division

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

# analysis/multisensory
multisensory_analysisfile = os.path.join(analysispath, 'multisensory.py')
multisensory_analysis     = imp.load_source('multisensory_analysis',
                                            multisensory_analysisfile)

# models/multisensory
multisensory_modelfile = os.path.join(modelspath, 'multisensory.py')
multisensory_model     = imp.load_source('multisensory_model', multisensory_modelfile)
multisensory_behavior  = os.path.join(trialspath, 'multisensory', 'trials_behavior.pkl')
multisensory_activity  = os.path.join(trialspath, 'multisensory', 'trials_activity.pkl')

#=========================================================================================

fig  = Figure()
plot = fig.add()

sigmas = []
for s in [''] + ['_s'+str(i) for i in xrange(101, 106)]:
    behaviorfile = os.path.join(trialspath, 'multisensory'+s, 'trials_behavior.pkl')
    sigmas.append(multisensory_analysis.psychometric(behaviorfile, plot))

fig.save()

#=========================================================================================

print("")
for i, (sigma_v, sigma_a, sigma_va) in enumerate(sigmas):
    if i == 0:
        print(r"\textbf{{{:.3f}}} & \textbf{{{:.3f}}} & \textbf{{{:.3f}}} & \textbf{{{:.3f}}} & \textbf{{{:.3f}}} \\"
              .format(sigma_v, sigma_a, sigma_va, 1/sigma_v**2 + 1/sigma_a**2, 1/sigma_va**2))
    else:
        print("{:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\"
              .format(sigma_v, sigma_a, sigma_va, 1/sigma_v**2 + 1/sigma_a**2, 1/sigma_va**2))
print("")
