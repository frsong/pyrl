from __future__ import absolute_import, division

import os

import numpy as np

from pyrl          import runtools, tasktools, utils
from pyrl.figtools import Figure, mpl

#/////////////////////////////////////////////////////////////////////////////////////////

cmap = mpl.cm.jet
norm = mpl.colors.Normalize(vmin=10, vmax=34)
smap = mpl.cm.ScalarMappable(norm, cmap)

#/////////////////////////////////////////////////////////////////////////////////////////

def performance(trialsfile, plot, **kwargs):
    # Load trials
    trials, A, R, M, perf = utils.load(trialsfile)

    correct_by_cond = {}
    for n, trial in enumerate(trials):
        if not perf.decisions[n]:
            continue

        gt_lt = trial['gt_lt']
        fpair = trial['fpair']
        if gt_lt == '>':
            f1, f2 = fpair
        else:
            f2, f1 = fpair
        cond = (f1, f2)

        correct_by_cond.setdefault(cond, []).append(perf.corrects[n])

    pcorrect_by_cond = {}
    for c in correct_by_cond:
        corrects = correct_by_cond[c]
        pcorrect_by_cond[c] = utils.divide(sum(corrects), len(corrects))

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    plot.equal()

    lw         = kwargs.get('lw', 1)
    fontsize   = kwargs.get('fontsize', 10)
    _min, _max = kwargs.get('lims', (10-4, 34+4))
    r          = kwargs.get('r', 1.5)

    for (f1, f2), pcorrect in pcorrect_by_cond.items():
        plot.circle((f1, f2), r, ec='none', fc=smap.to_rgba(f1))
        plot.text(f1, f2, '{}'.format(int(100*pcorrect)), color='w', fontsize=fontsize,
                  ha='center', va='center')

    plot.xlim(_min, _max)
    plot.ylim(_min, _max)
    plot.plot([_min, _max], [_min, _max], color='k', lw=lw)

def sort(trialsfile, plots, units=None, network='p', **kwargs):
    """
    Sort trials.

    """
    # Load trials
    data = utils.load(trialsfile)
    if len(data) == 9:
        trials, U, Z, A, P, M, perf, r_p, r_v = data
    else:
        trials, U, Z, Z_b, A, P, M, perf, r_p, r_v = data

    # Which network?
    if network == 'p':
        r = r_p
    else:
        r = r_v

    # Data shape
    Ntime = r.shape[0]
    N     = r.shape[-1]

    # Same for every trial
    time = trials[0]['time']

    # Aligned time
    time_a  = np.concatenate((-time[1:][::-1], time))
    Ntime_a = len(time_a)

    #=====================================================================================
    # Sort trials
    #=====================================================================================

    # Sort
    trials_by_cond = {}
    for n, trial in enumerate(trials):
        if perf.choices[n] is None or not perf.corrects[n]:
            continue

        # Condition
        gt_lt = trial['gt_lt']
        fpair = trial['fpair']
        if gt_lt == '>':
            f1, f2 = fpair
        else:
            f2, f1 = fpair
        cond = (f1, f2)

        # Firing rates
        Mn = np.tile(M[:,n], (N,1)).T
        Rn = r[:,n]*Mn

        # Align point
        t0 = trial['epochs']['f1'][0] - 1

        # Storage
        trials_by_cond.setdefault(cond, {'r': np.zeros((Ntime_a, N)),
                                         'n': np.zeros((Ntime_a, N), dtype=int)})

        # Before
        n_b = Rn[:t0].shape[0]
        trials_by_cond[cond]['r'][Ntime-1-n_b:Ntime-1] += Rn[:t0]
        trials_by_cond[cond]['n'][Ntime-1-n_b:Ntime-1] += Mn[:t0]

        # After
        n_a = Rn[t0:].shape[0]
        trials_by_cond[cond]['r'][Ntime-1:Ntime-1+n_a] += Rn[t0:]
        trials_by_cond[cond]['n'][Ntime-1:Ntime-1+n_a] += Mn[t0:]

    # Average
    for cond in trials_by_cond:
        trials_by_cond[cond] = utils.div(trials_by_cond[cond]['r'],
                                         trials_by_cond[cond]['n'])

    #=====================================================================================
    # Plot
    #=====================================================================================

    lw = kwargs.get('lw', 1.5)

    w, = np.where((time_a >= -500) & (time_a <= 4000))
    def plot_sorted(plot, unit):
        t    = 1e-3*time_a[w]
        yall = [[1]]
        for (f1, f2), r in trials_by_cond.items():
            plot.plot(t, r[w,unit], color=smap.to_rgba(f1), lw=lw)
            yall.append(r[w,unit])

        return t, yall

    if units is not None:
        for plot, unit in zip(plots, units):
            plot_sorted(plot, unit)
    else:
        figspath, name = plots
        for unit in xrange(N):
            fig  = Figure()
            plot = fig.add()

            #-----------------------------------------------------------------------------

            t, yall = plot_sorted(plot, unit)

            plot.xlim(t[0], t[-1])
            plot.lim('y', yall, lower=0)

            plot.highlight(0, 0.5)
            plot.highlight(3.5, 4)

            #-----------------------------------------------------------------------------

            fig.save(path=figspath, name=name+'_{}{:03d}'.format(network, unit))
            fig.close()

#/////////////////////////////////////////////////////////////////////////////////////////

def do(action, args, config):
    """
    Manage tasks.

    """
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    if 'trials' in action:
        try:
            trials_per_condition = int(args[0])
        except:
            trials_per_condition = 100

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'], config['dt'])

        spec         = model.spec
        gt_lts       = spec.gt_lts
        fpairs       = spec.fpairs
        n_conditions = spec.n_conditions
        n_trials     = trials_per_condition * n_conditions

        print("{} trials".format(n_trials))
        task   = model.Task()
        trials = []
        for n in xrange(n_trials):
            k = tasktools.unravel_index(n, (len(gt_lts), len(fpairs)))
            context = {
                'delay': 3000,
                'gt_lt': gt_lts[k.pop(0)],
                'fpair': fpairs[k.pop(0)]
                }
            trials.append(task.get_condition(pg.rng, pg.dt, context))
        runtools.run(action, trials, pg, config['trialspath'])

    #=====================================================================================

    elif action == 'performance':
        trialsfile = runtools.behaviorfile(config['trialspath'])

        fig  = Figure()
        plot = fig.add()

        performance(trialsfile, plot)

        plot.xlabel('$f_1$ (Hz)')
        plot.ylabel('$f_2$ (Hz)')

        fig.save(os.path.join(config['figspath'], action))

    #=====================================================================================

    elif action == 'sort':
        if 'value' in args:
            network = 'v'
        else:
            network = 'p'

        trialsfile = runtools.activityfile(config['trialspath'])
        sort(trialsfile, (config['figspath'], 'sorted'), network=network)
