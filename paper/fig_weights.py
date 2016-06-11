import imp
import os
import sys

import numpy as np

from pyrl          import matrixtools, utils
from pyrl.figtools import Figure

#=========================================================================================
# Files
#=========================================================================================

here   = utils.get_here(__file__)
parent = utils.get_parent(here)

paperpath = os.path.join(parent, 'paper')
timespath = os.path.join(paperpath, 'times')
figspath  = os.path.join(paperpath, 'work', 'figs')

modelname = sys.argv[1]

#=========================================================================================
# Figure
#=========================================================================================

w   = utils.mm_to_inch(174)
r   = 0.48
fig = Figure(w=w, r=r, axislabelsize=11, labelpadx=6, labelpady=6,
             thickness=0.9, ticksize=5, ticklabelsize=9, ticklabelpad=3)

x0 = 0.11
y0 = 0.18

w = 0.24
h = 0.71

DX = 0.07

fig.add('Wrec',        [x0, y0, w, h])
fig.add('Wrec_lambda', [fig[-1].right+DX, y0, w, h])
fig.add('Wrec_gamma',  [fig[-1].right+DX, y0, w, h])

#=========================================================================================

datapath = os.path.join(parent, 'examples', 'work', 'data', modelname)
savefile = os.path.join(datapath, modelname+'.pkl')
save     = utils.load(savefile)

params = save['best_policy_params']
N      = params['Wrec'].shape[0]

def plot_weights(plot, W):
    w     = np.ravel(W)
    w_exc = w[np.where(w > 0)]
    w_inh = w[np.where(w < 0)]

    plot.hist(w_exc, color=Figure.colors('blue'))
    plot.hist(w_inh, color=Figure.colors('red'))

fontsize = 9

plot = fig['Wrec']
W   = params['Wrec']
rho = matrixtools.spectral_radius(W)
#plot_weights(plot, W)
plot.text_upper_left(r'$W_\text{rec}$', fontsize=fontsize)
plot.text_upper_right(r'$\rho={:.3f}$'.format(rho), fontsize=fontsize)
plot.xlabel('$W$')

print(W[:10,:10])
Wnz = W[np.where(W != 0)]
print(len(Wnz))
exit()

plot = fig['Wrec_lambda']
W   = params['Wrec_gates'][:,:N]
rho = matrixtools.spectral_radius(W)
plot_weights(plot, W)
plot.text_upper_left(r'$W_\text{rec}^\lambda$', fontsize=fontsize)
plot.text_upper_right(r'$\rho={:.3f}$'.format(rho), fontsize=fontsize)

plot = fig['Wrec_gamma']
W   = params['Wrec_gates'][:,N:]
rho = matrixtools.spectral_radius(W)
plot_weights(plot, W)
plot.text_upper_left(r'$W_\text{rec}^\gamma$', fontsize=fontsize)
plot.text_upper_right(r'$\rho={:.3f}$'.format(rho), fontsize=fontsize)

#=========================================================================================

fig.save(path=figspath, name='fig_weights_'+modelname)
