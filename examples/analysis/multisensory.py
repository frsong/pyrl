from __future__ import absolute_import, division

import os

import numpy as np

from pyrl          import fittools, runtools, tasktools, utils
from pyrl.figtools import Figure

#/////////////////////////////////////////////////////////////////////////////////////////

colors = {
    'v':  Figure.colors('blue'),
    'a':  Figure.colors('green'),
    'va': Figure.colors('orange')
    }

#/////////////////////////////////////////////////////////////////////////////////////////

def psychometric(trialsfile, plot, **kwargs):
    # Load trials
    trials, A, R, M, perf = utils.load(trialsfile)

    decision_by_freq = {}
    high_by_freq     = {}
    for n, trial in enumerate(trials):
        mod  = trial['mod']
        freq = trial['freq']
        decision_by_freq.setdefault(mod, {})
        high_by_freq.setdefault(mod, {})
        decision_by_freq[mod].setdefault(freq, [])
        high_by_freq[mod].setdefault(freq, [])
        if perf.decisions[n]:
            decision_by_freq[mod][freq].append(True)

            if perf.choices[n] == 'H':
                high = 1
            else:
                high = 0

            high_by_freq[mod][freq].append(high)
        else:
            decision_by_freq[mod][freq].append(False)

    freqs      = {}
    p_decision = {}
    p_high     = {}
    for mod in decision_by_freq:
        freqs[mod]      = np.sort(high_by_freq[mod].keys())
        p_decision[mod] = np.zeros(len(freqs[mod]))
        p_high[mod]     = np.zeros(len(freqs[mod]))
        for i, freq in enumerate(freqs[mod]):
            p_decision[mod][i] = sum(decision_by_freq[mod][freq])/len(decision_by_freq[mod][freq])
            p_high[mod][i]     = utils.divide(sum(high_by_freq[mod][freq]), len(high_by_freq[mod][freq]))

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    lw = kwargs.get('lw', 1.5)
    ms = kwargs.get('ms', 6)

    all_x  = []
    all_y  = []
    sigmas = {}
    for mod in ['v', 'a', 'va']:
        if mod == 'v':
            label = 'Visual'
        elif mod == 'a':
            label = 'Auditory'
        elif mod == 'va':
            label = 'Multisensory'
        else:
            raise ValueError

        x = freqs[mod]
        y = p_high[mod]

        # Fit psychometric curve
        props = dict(lw=lw, color=colors[mod], label=label)
        try:
            popt, func = fittools.fit_psychometric(x, y)
            sigmas[mod] = popt['sigma']

            fit_x = np.linspace(min(x), max(x), 201)
            fit_y = func(fit_x, **popt)
            plot.plot(fit_x, 100*fit_y, **props)
        except RuntimeError:
            print("Unable to fit, drawing a line through the points.")
            plot.plot(x, 100*y, **props)
        plot.plot(x, 100*y, 'o', ms=ms, mew=0, mfc=props['color'])
        all_x.append(x)

    # Is it optimal?
    print("")
    print("  Optimality test")
    print("  ---------------")
    print("")
    for mod in ['v', 'a', 'va']:
        print("  sigma_{:<2} = {:.6f}".format(mod, sigmas[mod]))
    print("  1/sigma_v**2 + 1/sigma_a**2 = {:.6f}"
          .format(1/sigmas['v']**2 + 1/sigmas['a']**2))
    print("  1/sigma_va**2               = {:.6f}".format(1/sigmas['va']**2))

    plot.xlim(np.min(all_x), np.max(all_x))
    plot.ylim(0, 100)
    plot.yticks([0, 50, 100])

    plot.xlabel('Frequency (events/sec)')
    plot.ylabel('Percent high')

#/////////////////////////////////////////////////////////////////////////////////////////

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

    # Number of units
    N = r.shape[-1]

    # Same for every trial
    time  = trials[0]['time']
    Ntime = len(time)

    # Aligned time
    time_a  = np.concatenate((-time[1:][::-1], time))
    Ntime_a = len(time_a)

    #=====================================================================================
    # Aligned to stimulus onset
    #=====================================================================================

    r_by_cond_stimulus   = {}
    n_r_by_cond_stimulus = {}
    for n, trial in enumerate(trials):
        if not perf.decisions[n]:
            continue

        if trial['mod'] == 'va':
            continue
        assert trial['mod'] == 'v' or trial['mod'] == 'a'

        if not perf.corrects[n]:
            continue

        # Condition
        mod    = trial['mod']
        choice = perf.choices[n]
        cond   = (mod, choice)

        # Storage
        r_by_cond_stimulus.setdefault(cond, np.zeros((Ntime_a, N)))
        n_r_by_cond_stimulus.setdefault(cond, np.zeros((Ntime_a, N)))

        # Firing rates
        Mn = np.tile(M[:,n], (N,1)).T
        Rn = r[:,n]*Mn

        # Align point
        t0 = trial['epochs']['stimulus'][0] - 1

        # Before
        n_b = Rn[:t0].shape[0]
        r_by_cond_stimulus[cond][Ntime-1-n_b:Ntime-1]   += Rn[:t0]
        n_r_by_cond_stimulus[cond][Ntime-1-n_b:Ntime-1] += Mn[:t0]

        # After
        n_a = Rn[t0:].shape[0]
        r_by_cond_stimulus[cond][Ntime-1:Ntime-1+n_a]   += Rn[t0:]
        n_r_by_cond_stimulus[cond][Ntime-1:Ntime-1+n_a] += Mn[t0:]

    for cond in r_by_cond_stimulus:
        r_by_cond_stimulus[cond] = utils.div(r_by_cond_stimulus[cond],
                                             n_r_by_cond_stimulus[cond])

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    lw     = kwargs.get('lw', 1.5)
    dashes = kwargs.get('dashes', [3, 2])

    vline_props = {'lw': kwargs.get('lw_vline', 0.5)}
    if 'dashes_vline' in kwargs:
        vline_props['linestyle'] = '--'
        vline_props['dashes']    = dashes

    colors_by_mod = {
        'v': Figure.colors('blue'),
        'a': Figure.colors('green')
        }
    linestyle_by_choice = {
        'L': '-',
        'H': '--'
        }
    lineprops = dict(lw=lw)

    def plot_sorted(plot, unit, w, r_sorted):
        t = time_a[w]
        yall = [[1]]
        for cond in [('v', 'H'), ('v', 'L'), ('a', 'H'), ('a', 'L')]:
            mod, choice = cond

            if mod == 'v':
                label = 'Vis, '
            elif mod == 'a':
                label = 'Aud, '
            else:
                raise ValueError(mod)

            if choice == 'H':
                label += 'high'
            elif choice == 'L':
                label += 'low'
            else:
                raise ValueError(choice)

            linestyle = linestyle_by_choice[choice]
            if linestyle == '-':
                lineprops = dict(linestyle=linestyle, lw=lw)
            else:
                lineprops = dict(linestyle=linestyle, lw=lw, dashes=dashes)
            plot.plot(t, r_sorted[cond][w,unit],
                      color=colors_by_mod[mod],
                      label=label,
                      **lineprops)
            yall.append(r_sorted[cond][w,unit])

        return t, yall

    def on_stimulus(plot, unit):
        w, = np.where((time_a >= -300) & (time_a <= 1000))
        t, yall = plot_sorted(plot, unit, w, r_by_cond_stimulus)

        plot.xlim(t[0], t[-1])

        return yall

    if units is not None:
        for plot, unit in zip(plots, units):
            on_stimulus(plot, unit)
    else:
        figspath, name = plots
        for unit in xrange(N):
            fig  = Figure()
            plot = fig.add()

            #-----------------------------------------------------------------------------

            yall = []
            yall += on_stimulus(plot, unit)

            plot.lim('y', yall, lower=0)
            plot.vline(0)

            plot.xlabel('Time (ms)')
            plot.ylabel('Firing rate (a.u.)')

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

    #=====================================================================================

    if 'trials' in action:
        try:
            trials_per_condition = int(args[0])
        except:
            trials_per_condition = 500

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'], config['dt'])

        # Conditions
        spec         = model.spec
        mods         = spec.mods
        freqs        = spec.freqs
        n_conditions = spec.n_conditions
        n_trials     = n_conditions * trials_per_condition

        print("{} trials".format(n_trials))
        task   = model.Task()
        trials = []
        for n in xrange(n_trials):
            k       = tasktools.unravel_index(n, (len(mods), len(freqs)))
            context = {'mods': mods[k.pop(0)], 'freqs': freqs[k.pop(0)]}
            trials.append(task.get_condition(pg.rng, pg.dt, context))
        runtools.run(action, trials, pg, config['trialspath'])

    #=====================================================================================

    elif action == 'psychometric':
        trialsfile = runtools.behaviorfile(config['trialspath'])

        fig  = Figure()
        plot = fig.add()

        psychometric(trialsfile, plot)

        plot.vline(config['model'].spec.boundary)

        fig.save(path=config['figspath'], name='psychometric')
        fig.close()

    #=====================================================================================

    elif action == 'sort':
        if 'value' in args:
            network = 'v'
        else:
            network = 'p'

        trialsfile = runtools.activityfile(config['trialspath'])
        sort(trialsfile, (config['figspath'], 'sorted'), network=network)
