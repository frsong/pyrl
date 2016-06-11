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
p.add_argument('args', nargs='*')
a = p.parse_args()

simulate = a.simulate
args     = a.args
if not args:
    args = [
        'mante', # Fig. 1
        ]

#=========================================================================================
# Shared steps
#=========================================================================================

here   = utils.get_here(__file__)
parent = utils.get_parent(here)

dopath       = join(parent, 'examples')
modelspath   = join(parent, 'examples', 'models')
analysispath = join(parent, 'examples', 'analysis')
paperpath    = join(parent, 'paper')
timespath    = join(paperpath, 'times')
#paperfigspath = join(paperpath, 'work', 'figs')

# Make paths
#utils.mkdir_p(paperfigspath)
utils.mkdir_p(timespath)

def call(s):
    if simulate:
        print(3*' ' + s)
    else:
        rval = subprocess.call(s.split())
        if rval != 0:
            sys.stdout.flush()
            print("Something went wrong (return code {}).".format(rval))
            sys.exit(1)

def clean(model):
    call("python {} {} clean"
         .format(join(dopath, 'do.py'), join(modelspath, model)))

def train(model, seed=None, extra=''):
    if seed is None:
        extra  = ''
        suffix = ''
    else:
        extra = ' --seed {0} --suffix _s{0}'.format(seed)
        suffix = '_s'+str(seed)

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

def do_action(model, action, analysis=None, args=''):
    if analysis is None:
        analysis = model.split('_')[0]

    call("python {} {} run {} {} {}".format(join(dopath, 'do.py'),
                                            join(modelspath, model),
                                            join(analysispath, analysis),
                                            action,
                                            args))

def trials(trialtype, model, ntrials, analysis=None, args=''):
    do_action(model, 'trials-{} {}'.format(trialtype, ntrials),
              analysis=analysis, args=args)

def figure(fig):
    call('python ' + join(paperpath, fig + '.py'))

#=========================================================================================

#-----------------------------------------------------------------------------------------
# RDM (FD)
#-----------------------------------------------------------------------------------------

if 'rdm_fixed' in args:
    print("=> Perceptual decision-making (FD)")
    #train('rdm_fixed')
    #trials('b', 'rdm_fixed', 2500)
    #do_action('rdm_fixed', 'correct_stimulus_duration')
    trials('a', 'rdm_fixed', 200)
    #do_action('rdm_fixed', 'sort')

if 'rdm_fixed-seeds' in args:
    print("=> Perceptual decision-making (FD) (additional)")
    train_seeds('rdm_fixed', n_train=5)

#-----------------------------------------------------------------------------------------

if 'rdm_rt' in args:
    print("=> Perceptual decision-making (RT)")

    #trials('b', 'rdm_rt', 200, analysis='rdm')
    #do_action('rdm_rt', 'psychometric')
    #do_action('rdm_rt', 'chronometric')

    trials('a', 'rdm_rt', 100)
    do_action('rdm_rt', 'sort')#, args='value')
    #do_action('rdm_rt', 'sort', args='value')

#-----------------------------------------------------------------------------------------
# Context-dependent integration
#-----------------------------------------------------------------------------------------

if 'mante' in args:
    print("=> Context-dependent integration")
    #clean('mante')
    train('mante')
    trials('b', 'mante', 100)
    do_action('mante', 'psychometric')
    trials('a', 'mante', 20)
    do_action('mante', 'sort')

    #do_action('mante', 'regress')
    #do_action('mante', 'units')
    #figure('fig_mante')

if 'mante-seeds' in args:
    print("=> Context-dependent integration (additional)")
    train_seeds('mante', n_train=5)

#-----------------------------------------------------------------------------------------
# Multisensory integration
#-----------------------------------------------------------------------------------------

if 'multisensory' in args:
    print("=> Multisensory integration")
    #train('multisensory')
    trials('b', 'multisensory', 1500)
    do_action('multisensory', 'psychometric')
    #trials('a', 'multisensory', 100)
    #do_action('multisensory', 'sort')

if 'multisensory-seeds' in args:
    print("=> Multisensory integration (additional)")
    train_seeds('multisensory', n_train=5)

#-----------------------------------------------------------------------------------------
# Parametric working memory
#-----------------------------------------------------------------------------------------

if 'romo' in args:
    print("=> Parametric working memory")
    #train('romo')
    trials('b', 'romo', 100)
    do_action('romo', 'performance')
    trials('a', 'romo', 20)
    do_action('romo', 'sort')

if 'romo-seeds' in args:
    print("=> Parametric working memory (additional)")
    train_seeds('romo', n_train=5)

#-----------------------------------------------------------------------------------------
# Postdecision wager
#-----------------------------------------------------------------------------------------

# postdecisionager2
# --seed 1001
if 'postdecisionwager' in args:
    print("=> Postdecision wagering")
    #train('postdecisionwager2')
    #trials('b', 'postdecisionwager', 20, analysis='postdecisionwager'); exit()
    trials('b', 'postdecisionwager', 2500, analysis='postdecisionwager')
    do_action('postdecisionwager', 'sure_stimulus_duration', analysis='postdecisionwager')
    do_action('postdecisionwager', 'correct_stimulus_duration', analysis='postdecisionwager')
    trials('a', 'postdecisionwager', 100, analysis='postdecisionwager')
    do_action('postdecisionwager', 'sort', analysis='postdecisionwager')

if 'postdecisionwager-seeds' in args:
    print("=> Postdecision wagering (additional)")
    train_seeds('postdecisionwager', n_train=5)

#-----------------------------------------------------------------------------------------

if 'padoaschioppa2006' in args:
    print("=> Padoa-Schioppa 2006")
    #train('padoaschioppa2006')
    trials('b', 'padoaschioppa2006', 200)
    do_action('padoaschioppa2006', 'choice_pattern')
    trials('a', 'padoaschioppa2006', 200)
    do_action('padoaschioppa2006', 'sort_epoch', args='postoffer value')
    do_action('padoaschioppa2006', 'sort_epoch', args='latedelay value')
    do_action('padoaschioppa2006', 'sort_epoch', args='prechoice value')
    do_action('padoaschioppa2006', 'sort_epoch', args='prechoice value separate-by-choice')

if 'padoaschioppa2006-seeds' in args:
    print("=> Padoa-Schioppa 2006 (additional)")
    train_seeds('padoaschioppa2006', n_train=5)

if 'padoaschioppa2006-1A3B' in args:
    trials('b', 'padoaschioppa2006_1A3B', 200)
    do_action('padoaschioppa2006_1A3B', 'choice_pattern')
