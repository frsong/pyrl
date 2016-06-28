#!/usr/bin/env python
"""
Script for performing common tasks.

"""
import argparse
import imp
import os
import shutil
import sys
import time

#=========================================================================================
# Command line
#=========================================================================================

p = argparse.ArgumentParser()
p.add_argument('model_file', help="model specification")
p.add_argument('action', nargs='?', type=str, default='info')
p.add_argument('args', nargs='*')
p.add_argument('--dt', type=float, default=0)
p.add_argument('--dt-save', type=float, default=0)
p.add_argument('--seed', type=int, default=100)
p.add_argument('--suffix', type=str, default='')
p.add_argument('--gpu', dest='gpu', action='store_true', default=False)
a = p.parse_args()

# Model file
modelfile = os.path.abspath(a.model_file)
if not modelfile.endswith('.py'):
    modelfile += '.py'

action  = a.action
args    = a.args
dt      = a.dt
dt_save = a.dt_save
seed    = a.seed
suffix  = a.suffix
gpu     = a.gpu

print("MODELFILE: " + modelfile)
print("ACTION:    " + action)
print("ARGS:      " + str(args))
print("SEED:      " + str(seed))
print("SUFFIX:    " + suffix)
print("GPU:       " + str(gpu))

# Set flags before importing Theano
os.environ.setdefault('THEANO_FLAGS', '')
os.environ['THEANO_FLAGS'] += ',floatX=float32,allow_gc=False'
if gpu:
    os.environ['THEANO_FLAGS'] += ',device=gpu,nvcc.fastmath=True'

from pyrl       import utils
from pyrl.model import Model

#=========================================================================================
# Setup paths
#=========================================================================================

# Location of script
here   = utils.get_here(__file__)
prefix = os.path.basename(here)

# Name to use
name = os.path.splitext(os.path.basename(modelfile))[0] + suffix

# Scratch
scratchpath = os.environ.get('SCRATCH')
if scratchpath is None:
    scratchpath = os.path.join(os.environ['HOME'], 'scratch')
trialspath = os.path.join(scratchpath, 'work', 'pyrl', prefix, name)

# Paths
workpath = os.path.join(here,     'work')
datapath = os.path.join(workpath, 'data', name)
figspath = os.path.join(workpath, 'figs', name)

# Create necessary directories
for path in [datapath, figspath, trialspath]:
    utils.mkdir_p(path)

# File to store model in
savefile = os.path.join(datapath, name + '.pkl')

#=========================================================================================
# Info
#=========================================================================================

if action == 'info':
    # Model specification
    model = Model(modelfile)

    # Create a PolicyGradient instance
    pg = model.get_pg(savefile, seed)

    # Print additional info
    config = pg.config
    #print(config.keys())
    #print("Seed (policy):   {}".format(config['policy_seed']))
    #print("Seed (baseline): {}".format(config['baseline_seed']))

#=========================================================================================
# Train
#=========================================================================================

elif action == 'train':
    # Model specification
    model = Model(modelfile)

    # Train
    model.train(savefile, seed, recover=('recover' in args))

#=========================================================================================
# Run analysis
#=========================================================================================

elif action == 'run':
    # Get analysis script
    try:
        runfile = args[0]
    except IndexError:
        print("Please specify the analysis script.")
        sys.exit()
    if not runfile.endswith('.py'):
        runfile += '.py'

    # Load analysis module
    try:
        r = imp.load_source('analysis', runfile)
    except IOError:
        print("Couldn't load analysis module from {}".format(runfile))
        sys.exit()

    # Load model
    model = Model(modelfile)

    # Reset args
    args = args[1:]
    if len(args) > 0:
        action = args[0]
        args   = args[1:]
    else:
        action = None
        args   = []

    # Copy the savefile for safe access
    if os.path.isfile(savefile):
        base, ext = os.path.splitext(savefile)
        savefile_copy = base + '_copy.pkl'
        while True:
            shutil.copy(savefile, savefile_copy)
            try:
                utils.load(savefile_copy)
                break
            except EOFError:
                continue
    else:
        print("File {} doesn't exist.".format(savefile))

    # Pass everything on
    config = {
        'seed':       1,
        'suffix':     suffix,
        'model':      model,
        'savefile':   savefile_copy,
        'datapath':   datapath,
        'figspath':   figspath,
        'trialspath': trialspath
        }

    if dt > 0:
        config['dt'] = dt
    else:
        config['dt'] = None

    if dt_save > 0:
        config['dt-save'] = dt_save
    else:
        config['dt-save'] = None

    try:
        r.do(action, args, config)
    except SystemExit as e:
        print("Error: " + str(e.code))
        raise

#=========================================================================================

else:
    print("Unrecognized action \'{}\'.".format(action))
