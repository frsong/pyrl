"""
Reproduce every figure in the paper from scratch.

"""
from __future__ import division

import argparse
import datetime
import os
import subprocess
import sys
from   os.path import join

import numpy as np

from pyrl import utils

#=========================================================================================
# Command line
#=========================================================================================

p = argparse.ArgumentParser()
p.add_argument('--simulate', action='store_true', default=False)
p.add_argument('--gpu', dest='gpu', action='store_true', default=False)
p.add_argument('args', nargs='*')
a = p.parse_args()

simulate = a.simulate
args     = a.args
gpu      = a.gpu

#=========================================================================================
# Shared steps
#=========================================================================================

here   = utils.get_here(__file__)
parent = utils.get_parent(here)

dopath        = join(parent, 'examples')
modelspath    = join(parent, 'examples', 'models')
analysispath  = join(parent, 'examples', 'analysis')
paperpath     = join(parent, 'paper')
timespath     = join(paperpath, 'times')
paperdatapath = join(paperpath, 'work', 'data')
paperfigspath = join(paperpath, 'work', 'figs')

# Make paths
for path in [timespath, paperdatapath, paperfigspath]:
    utils.mkdir_p(path)

def call(s):
    if simulate:
        print(3*' ' + s)
    else:
        rval = subprocess.call(s.split())
        if rval != 0:
            sys.stdout.flush()
            print("Something went wrong (return code {}).".format(rval))
            sys.exit(1)

def train(model, seed=None, main=False):
    if seed is None:
        extra = ''
    else:
        extra = ' --seed ' + str(seed)

    if seed is None or main:
        suffix = ''
    else:
        suffix  = '_s' + str(seed)
        extra  += ' --suffix ' + suffix

    if gpu:
        extra += ' --gpu'

    tstart = datetime.datetime.now()
    call("python {} {} train{}".format(join(dopath, 'do.py'),
                                       join(modelspath, model),
                                       extra))
    tend = datetime.datetime.now()

    # Save training time
    totalmins = int((tend - tstart).total_seconds()/60)
    timefile = join(timespath, model + suffix + '.txt')
    np.savetxt(timefile, [totalmins], fmt='%d', header='mins')

def train_seeds(model, start_seed=1000, n_train=1):
    for seed in xrange(start_seed, start_seed+n_train):
        print("[ train_seeds ] {}".format(seed))
        train(model, seed=seed)

def do_action(model, action, analysis=None, seed=None, args=''):
    if analysis is None:
        analysis = model.split('_')[0]

    if seed is not None:
        args = '--suffix _s{0} '.format(seed) + args

    call("python {} {} run {} {} {}".format(join(dopath, 'do.py'),
                                            join(modelspath, model),
                                            join(analysispath, analysis),
                                            action,
                                            args))

def trials(model, trialtype, ntrials, analysis=None, seed=None, args=''):
    do_action(model, 'trials-{} {}'.format(trialtype, ntrials),
              analysis=analysis, seed=seed, args=args)

def figure(fig, args=''):
    call('python {} {}'.format(join(paperpath, fig + '.py'), args))

#=========================================================================================
# Tasks
#=========================================================================================

start_seed = 101
ntrain     = 5

#-----------------------------------------------------------------------------------------
# RDM (FD)
#-----------------------------------------------------------------------------------------

model     = 'rdm_fixed'
ntrials_b = 5000
ntrials_a = 50

if 'rdm_fixed' in args:
    print("=> Perceptual decision-making (FD)")
    seed = 97
    train(model, seed=seed, main=True)
    trials(model, 'b', ntrials_b)
    do_action(model, 'psychometric')
    do_action(model, 'correct_stimulus_duration')
    trials(model, 'a', ntrials_a)
    do_action(model, 'sort')

if 'rdm_fixed-seeds' in args:
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TRAIN: Perceptual decision-making (FD) (seed = {})".format(seed))
        train(model, seed=seed)
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TEST: Perceptual decision-making (FD) (seed = {})".format(seed))
        trials(model, 'b', ntrials_b, seed=seed)
        do_action(model, 'psychometric', seed=seed)
        do_action(model, 'correct_stimulus_duration', seed=seed)

model = 'rdm_fixedlinearbaseline'

if 'rdm_fixedlinearbaseline' in args:
    print("=> Perceptual decision-making (FD), linear baseline")
    seed = 97
    train(model, seed=seed, main=True)
    trials(model, 'b', ntrials_b)
    do_action(model, 'psychometric')
    do_action(model, 'correct_stimulus_duration')
    trials(model, 'a', ntrials_a)
    do_action(model, 'sort')

if 'rdm_fixedlinearbaseline-seeds' in args:
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TRAIN: Perceptual decision-making (FD), linear baseline (seed = {})".format(seed))
        train(model, seed=seed)
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TEST: Perceptual decision-making (FD), linear baseline (seed = {})".format(seed))
        trials(model, 'b', ntrials_b, seed=seed)
        do_action(model, 'psychometric', seed=seed)
        do_action(model, 'correct_stimulus_duration', seed=seed)

#-----------------------------------------------------------------------------------------
# RDM (FD), but small dt
#-----------------------------------------------------------------------------------------

model     = 'rdm_fixed_dt'
ntrials_b = 1000

if 'rdm_fixed_dt' in args:
    print("=> Perceptual decision-making (FD), dt = 1 ms")
    train(model)
    trials(model, 'b', ntrials_b, analysis='rdm', args='--dt-save 10')
    do_action(model, 'psychometric', analysis='rdm')
    do_action(model, 'correct_stimulus_duration', analysis='rdm')

#-----------------------------------------------------------------------------------------
# RDM (RT)
#-----------------------------------------------------------------------------------------

model     = 'rdm_rt'
ntrials_b = 2500
ntrials_a = 50

if 'rdm_rt' in args:
    print("=> Perceptual decision-making (RT)")
    train(model)
    trials(model, 'b', ntrials_b)
    do_action(model, 'psychometric')
    do_action(model, 'chronometric')
    trials(model, 'a', ntrials_a)
    do_action(model, 'sort')

if 'rdm_rt-seeds' in args:
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TRAIN: Perceptual decision-making (RT) (seed = {})".format(seed))
        train(model, seed=seed)
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TEST: Perceptual decision-making (RT) (seed = {})".format(seed))
        trials(model, 'b', ntrials_b, seed=seed)
        do_action(model, 'psychometric', seed=seed)
        do_action(model, 'chronometric', seed=seed)

#-----------------------------------------------------------------------------------------
# Context-dependent integration
#-----------------------------------------------------------------------------------------

model     = 'mante'
ntrials_b = 1000
ntrials_a = 100

if 'mante' in args:
    print("=> Context-dependent integration")
    train('mante')
    trials(model, 'b', ntrials_b)
    do_action(model, 'psychometric')
    trials(model, 'a', ntrials_a)
    do_action(model, 'sort')

if 'mante-seeds' in args:
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TRAIN: Context-dependent integration (seed = {})".format(seed))
        train(model, seed=seed)
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TEST: Context-dependent integration (seed = {})".format(seed))
        trials(model, 'b', ntrials_b, seed=seed)
        do_action(model, 'psychometric', seed=seed)
    #for seed in xrange(start_seed, start_seed+ntrain):
    #    print("=> TEST: Context-dependent integration (seed = {})".format(seed))
    #    trials(model, 'a', ntrials_a, seed=seed)
    #    do_action(model, 'statespace', seed=seed)

#-----------------------------------------------------------------------------------------
# Multisensory integration
#-----------------------------------------------------------------------------------------

model     = 'multisensory'
ntrials_b = 1500
ntrials_a = 100

if 'multisensory' in args:
    print("=> Multisensory integration")
    seed = 99
    train(model, seed=seed, main=True)
    trials(model, 'b', ntrials_b)
    do_action(model, 'psychometric')
    trials(model, 'a', ntrials_a)
    do_action(model, 'sort')

if 'multisensory-seeds' in args:
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TRAIN: Multisensory integration (seed = {})".format(seed))
        train(model, seed=seed)
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TEST: Multisensory integration (seed = {})".format(seed))
        trials(model, 'b', ntrials_b, seed=seed)
        do_action(model, 'psychometric', seed=seed)

#-----------------------------------------------------------------------------------------
# Parametric working memory
#-----------------------------------------------------------------------------------------

model     = 'romo'
ntrials_b = 100
ntrials_a = 50

if 'romo' in args:
    print("=> Parametric working memory")
    train('romo')
    trials(model, 'b', ntrials_b)
    do_action(model, 'performance')
    trials(model, 'a', ntrials_a)
    do_action(model, 'sort')

if 'romo-seeds' in args:
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TRAIN: Parametric working memory (seed = {})".format(seed))
        train(model, seed=seed)
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TEST: Parametric working memory (seed = {})".format(seed))
        trials(model, 'b', ntrials_b, seed=seed)
        do_action(model, 'performance', seed=seed)

#-----------------------------------------------------------------------------------------
# Postdecision wager
#-----------------------------------------------------------------------------------------

model     = 'postdecisionwager'
ntrials_b = 2500
ntrials_a = 100

if 'postdecisionwager' in args:
    print("=> Postdecision wager")
    train(model)
    trials(model, 'b', ntrials_b)
    do_action(model, 'sure_stimulus_duration')
    do_action(model, 'correct_stimulus_duration')
    trials(model, 'a', ntrials_a)
    do_action(model, 'sort')
    do_action(model, 'sort', args='value')

if 'postdecisionwager-seeds' in args:
    for seed in range(start_seed, start_seed+0) + [1000]:
        print("=> TRAIN: Postdecision wager (seed = {})".format(seed))
        train(model, seed=seed)
    for seed in range(start_seed, start_seed+0) + [1000]:
        print("=> TEST: Postdecision wager (seed = {})".format(seed))
        trials(model, 'b', ntrials_b, seed=seed)
        do_action(model, 'sure_stimulus_duration', seed=seed)
        do_action(model, 'correct_stimulus_duration', seed=seed)

model = 'postdecisionwager_linearbaseline'

if 'postdecisionwager_linearbaseline' in args:
    print("=> Postdecision wager, linear baseline")
    train(model)
    trials(model, 'b', ntrials_b)
    do_action(model, 'sure_stimulus_duration')
    do_action(model, 'correct_stimulus_duration')
    trials(model, 'a', ntrials_a)
    do_action(model, 'sort')
    do_action(model, 'sort', args='value')

if 'postdecisionwager_linearbaseline-seeds' in args:
    for seed in range(start_seed, start_seed+0) + [1000]:
        print("=> TRAIN: Postdecision wager, linear baseline (seed = {})".format(seed))
        train(model, seed=seed)
    for seed in range(start_seed, start_seed+0) + [1000]:
        print("=> TEST: Postdecision wager, linear baseline (seed = {})".format(seed))
        trials(model, 'b', ntrials_b, seed=seed)
        do_action(model, 'sure_stimulus_duration', seed=seed)
        do_action(model, 'correct_stimulus_duration', seed=seed)

#-----------------------------------------------------------------------------------------
# Economic choice
#-----------------------------------------------------------------------------------------

model     = 'padoaschioppa2006'
ntrials_b = 500
ntrials_a = 500

if 'padoaschioppa2006' in args:
    print("=> Padoa-Schioppa 2006")
    train('padoaschioppa2006')
    trials(model, 'b', ntrials_b)
    do_action(model, 'choice_pattern')
    do_action(model, 'indifference_point')
    trials(model, 'a', ntrials_a)
    do_action(model, 'sort_epoch', args='prechoice value')
    do_action(model, 'sort_epoch', args='prechoice value separate-by-choice')
    do_action(model, 'sort_epoch', args='prechoice policy')

if 'padoaschioppa2006-seeds' in args:
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TRAIN: Padoa-Schioppa 2006 (seed = {})".format(seed))
        train(model, seed=seed)
    for seed in xrange(start_seed, start_seed+ntrain):
        print("=> TEST: Padoa-Schioppa 2006 (seed = {})".format(seed))
        trials(model, 'b', ntrials_b, seed=seed)
        do_action(model, 'choice_pattern', seed=seed)
        do_action(model, 'indifference_point', seed=seed)

if 'padoaschioppa2006-1A3B' in args:
    train(model+'_1A3B')
    trials(model+'_1A3B', 'b', ntrials_b)
    do_action(model+'_1A3B', 'choice_pattern')
    do_action(model+'_1A3B', 'indifference_point')

#=========================================================================================
# Paper figures
#=========================================================================================

if 'fig1_rdm' in args:
    figure('fig1_rdm')

if 'fig_cognitive' in args:
    figure('fig_cognitive')

if 'fig_postdecisionwager' in args:
    figure('fig_postdecisionwager')

if 'fig_padoaschioppa2006' in args:
    figure('fig_padoaschioppa2006')

if 'fig_rdm_value' in args:
    figure('fig_rdm_value')

if 'fig_rdm_rt_value' in args:
    figure('fig_rdm_rt_value')

if 'fig_rdm_rt' in args:
    figure('fig_rdm_rt')

if 'fig-learning-rdm_fixed' in args:
    figure('fig_learning', args='rdm_fixed')

if 'fig-learning-rdm_fixedlinearbaseline' in args:
    figure('fig_learning', args='rdm_fixedlinearbaseline')

if 'fig-learning-rdm_rt' in args:
    figure('fig_learning', args='rdm_rt')

if 'fig-learning-mante' in args:
    figure('fig_learning', args='mante')

if 'fig-learning-multisensory' in args:
    figure('fig_learning', args='multisensory')

if 'fig-learning-romo' in args:
    figure('fig_learning', args='romo')

if 'fig-learning-postdecisionwager' in args:
    figure('fig_learning', args='postdecisionwager')

if 'fig-learning-padoaschioppa2006' in args:
    figure('fig_learning', args='padoaschioppa2006')
