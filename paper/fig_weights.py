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

masks  = save['policy_masks']
params = save['best_policy_params']
N      = params['Wrec'].shape[0]

Win = save['best_policy_params']['Win']
if 'Win' in save['policy_masks']:
    Win *= save['policy_masks']['Win']
print(np.mean(Win))
print(np.std(Win))

Win = save['best_baseline_params']['Win']
if 'Win' in save['baseline_masks']:
    Win *= save['baseline_masks']['Win']
print(np.mean(Win))
print(np.std(Win))

def plot_weights(plot, W):
    w     = np.ravel(W)
    w_exc = w[np.where(w > 0)]
    w_inh = w[np.where(w < 0)]

    plot.hist(w_exc, color=Figure.colors('blue'))
    plot.hist(w_inh, color=Figure.colors('red'))

fontsize = 9
dy       = 0.01

plot = fig['Wrec']
W = params['Wrec']
if 'Wrec' in masks:
    W *= masks['Wrec']

import matplotlib as mpl

plot = fig['Wrec']
#plot.imshow(W.T, cmap=mpl.cm.gray_r)
#'''
rho = matrixtools.spectral_radius(W)
plot_weights(plot, W)
plot.text_upper_left(r'$W_\text{rec}$', fontsize=fontsize, dy=dy)
plot.text_upper_right(r'$\rho={:.3f}$'.format(rho), fontsize=fontsize, dy=dy)
plot.xlabel('$W$')

#for j in xrange(W.shape[1]):
#    print(sum(1*(W[:,j] != 0)))

Wrec_gates = params['Wrec_gates']
if 'Wrec_gates' in masks:
    Wrec_gates *= masks['Wrec_gates']

plot = fig['Wrec_lambda']
W = Wrec_gates[:,:N]
rho = matrixtools.spectral_radius(W)
plot_weights(plot, W)
plot.text_upper_left(r'$W_\text{rec}^\lambda$', fontsize=fontsize, dy=dy)
plot.text_upper_right(r'$\rho={:.3f}$'.format(rho), fontsize=fontsize, dy=dy)

plot = fig['Wrec_gamma']
W = Wrec_gates[:,N:]
rho = matrixtools.spectral_radius(W)
plot_weights(plot, W)
plot.text_upper_left(r'$W_\text{rec}^\gamma$', fontsize=fontsize, dy=dy)
plot.text_upper_right(r'$\rho={:.3f}$'.format(rho), fontsize=fontsize, dy=dy)
#'''
#=========================================================================================

fig.save(path=figspath, name='fig_weights_'+modelname)
