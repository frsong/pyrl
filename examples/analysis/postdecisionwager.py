from __future__ import division

import os

import numpy as np

from pyrl          import datatools, fittools, runtools, tasktools, utils
from pyrl.figtools import Figure

#/////////////////////////////////////////////////////////////////////////////////////////

blues = {
    0:    '#c6dbef',
    3.2:  '#9ecae1',
    6.4:  '#6baed6',
    12.8: '#4292c6',
    25.6: '#2171b5',
    51.2: '#084594'
    }

kiani2009_colors = {
    0:    '#d53137',
    3.2:  '#9e9c3f',
    6.4:  '#1fa54d',
    12.8: '#2f377c',
    25.6: '#1a1919',
    51.2: '#d52f81'
    }

#/////////////////////////////////////////////////////////////////////////////////////////

def sure_stimulus_duration(trialsfile, plot, saved=None, **kwargs):
    if saved is not None:
        psure_by_duration_by_coh = saved
    else:
        trials, A, R, M, perf = utils.load(trialsfile)

        # Sort
        trials_by_cond = {}
        for n, trial in enumerate(trials):
            if not trial['wager']:
                continue

            cond = trial['coh']
            if perf.choices[n] is not None:
                trials_by_cond.setdefault(cond, {'durations': [], 'sures': []})

                duration = np.ptp(trial['durations']['stimulus'])
                trials_by_cond[cond]['durations'].append(duration)
                trials_by_cond[cond]['sures'].append(perf.choices[n] == 'S')

        # Number of bins
        nbins = kwargs.get('nbins', 10)

        # Average
        psure_by_duration_by_coh = {}
        for coh, v in trials_by_cond.items():
            (xbins, ybins, xedges,
             binsizes) = datatools.partition(v['durations'], v['sures'], nbins=nbins)
            duration = [np.mean(xbin) for xbin in xbins]
            psure = [utils.divide(np.sum(ybin > 0), len(ybin)) for ybin in ybins]
            psure_by_duration_by_coh[coh] = (duration, psure)

    #=====================================================================================
    # Plot
    #=====================================================================================

    lineprop = {'lw':  kwargs.get('lw', 1)}
    dataprop = {'ms':  kwargs.get('ms', 7),
                'mew': kwargs.get('mew', 0)}
    colors = kwargs.get('colors', kiani2009_colors)

    cohs = sorted(psure_by_duration_by_coh)
    for coh in cohs:
        duration, psure = psure_by_duration_by_coh[coh]

        plot.plot(duration, psure, color=colors[coh], label='{}\%'.format(coh), **lineprop)
        plot.plot(duration, psure, 'o', mfc=colors[coh], **dataprop)

    plot.xlim(100, 800)
    plot.ylim(0, 1)

    #=====================================================================================

    return psure_by_duration_by_coh

#/////////////////////////////////////////////////////////////////////////////////////////

def correct_stimulus_duration(trialsfile, plot, saved=None, **kwargs):
    if saved is not None:
        pcorrect_by_duration_by_coh, pcorrect_by_duration_by_coh_wager = saved
    else:
        trials, A, R, M, perf = utils.load(trialsfile)

        # Sort
        trials_by_cond       = {}
        trials_by_cond_wager = {}
        for n, trial in enumerate(trials):
            coh = trial['coh']
            if coh == 0 or perf.choices[n] not in ['L', 'R']:
                continue

            cond     = coh
            duration = np.ptp(trial['durations']['stimulus'])
            if trial['wager']:
                trials_by_cond_wager.setdefault(cond, {'durations': [], 'corrects': []})
                trials_by_cond_wager[cond]['durations'].append(duration)
                trials_by_cond_wager[cond]['corrects'].append(perf.corrects[n])
            else:
                trials_by_cond.setdefault(cond, {'durations': [], 'corrects': []})
                trials_by_cond[cond]['durations'].append(duration)
                trials_by_cond[cond]['corrects'].append(perf.corrects[n])

        # Number of bins
        nbins = kwargs.get('nbins', 10)

        # Average no-wager trials
        pcorrect_by_duration_by_coh = {}
        for coh, v in trials_by_cond.items():
            (xbins, ybins, xedges,
             binsizes) = datatools.partition(v['durations'], v['corrects'], nbins=nbins)
            duration = [np.mean(xbin) for xbin in xbins]
            pcorrect = [utils.divide(np.sum(ybin > 0), len(ybin)) for ybin in ybins]
            pcorrect_by_duration_by_coh[coh] = (duration, pcorrect)

        # Average wager trials
        pcorrect_by_duration_by_coh_wager = {}
        for coh, v in trials_by_cond_wager.items():
            (xbins, ybins, xedges,
             binsizes) = datatools.partition(v['durations'], v['corrects'], nbins=nbins)
            duration = [np.mean(xbin) for xbin in xbins]
            pcorrect = [utils.divide(np.sum(ybin > 0), len(ybin)) for ybin in ybins]
            pcorrect_by_duration_by_coh_wager[coh] = (duration, pcorrect)

    #=====================================================================================
    # Plot
    #=====================================================================================

    lineprop = {'ls':     '--',
                'lw':     kwargs.get('lw', 1),
                'dashes': kwargs.get('dashes', [9, 4])}
    dataprop = {'mew': kwargs.get('mew', 1)}
    dataprop['ms'] = kwargs.get('ms', 6) + dataprop['mew']/2

    lineprop_wager = {'lw':  kwargs.get('lw', 1)}
    dataprop_wager = {'ms':  kwargs.get('ms', 7),
                      'mew': kwargs.get('mew', 0)}

    colors = kwargs.get('colors', kiani2009_colors)

    # No-wager trials
    cohs = sorted(pcorrect_by_duration_by_coh)
    for coh in cohs:
        duration, pcorrect = pcorrect_by_duration_by_coh[coh]

        plot.plot(duration, pcorrect, color=colors[coh], zorder=10, **lineprop)
        plot.plot(duration, pcorrect, 'o', mfc='w', mec=colors[coh],
                  zorder=10, **dataprop)

    # Wager trials
    cohs = sorted(pcorrect_by_duration_by_coh_wager)
    for coh in cohs:
        duration, pcorrect = pcorrect_by_duration_by_coh_wager[coh]

        plot.plot(duration, pcorrect, color=colors[coh], zorder=5, **lineprop_wager)
        plot.plot(duration, pcorrect, 'o', mfc=colors[coh], mec=colors[coh],
                  zorder=5, **dataprop_wager)

    plot.xlim(100, 800)
    plot.ylim(0.5, 1)

    #=====================================================================================

    return pcorrect_by_duration_by_coh, pcorrect_by_duration_by_coh_wager

#/////////////////////////////////////////////////////////////////////////////////////////

def value_stimulus_duration(trialsfile, plot, saved=None, **kwargs):
    if saved is not None:
        value_by_duration_by_coh, value_by_duration_by_coh_wager = saved
    else:
        # Load trials
        trials, U, Z, Z_b, A, P, M, perf, r_p, r_v = utils.load(trialsfile)

        # Time
        time = trials[0]['time']

        # Sort
        trials_by_cond       = {}
        trials_by_cond_wager = {}
        for n, trial in enumerate(trials):
            coh = trial['coh']
            if coh == 0 or perf.choices[n] not in ['L', 'R']:
                continue

            cond     = coh
            duration = np.ptp(trial['durations']['stimulus'])

            delay_start, _ = trial['durations']['delay']
            before_sure,   = np.where((delay_start <= time) & (time < delay_start+500))
            value          = np.mean(Z_b[before_sure,n])
            if trial['wager']:
                trials_by_cond_wager.setdefault(cond, {'durations': [], 'values': []})
                trials_by_cond_wager[cond]['durations'].append(duration)
                trials_by_cond_wager[cond]['values'].append(value)
            else:
                trials_by_cond.setdefault(cond, {'durations': [], 'values': []})
                trials_by_cond[cond]['durations'].append(duration)
                trials_by_cond[cond]['values'].append(value)

        # Number of bins
        nbins = kwargs.get('nbins', 10)

        # Average no-wager trials
        value_by_duration_by_coh = {}
        for coh, v in trials_by_cond.items():
            (xbins, ybins, xedges,
             binsizes) = datatools.partition(v['durations'], v['values'], nbins=nbins)
            duration = [np.mean(xbin) for xbin in xbins]
            value    = [np.mean(ybin) for ybin in ybins]
            value_by_duration_by_coh[coh] = (duration, value)

        # Average wager trials
        value_by_duration_by_coh_wager = {}
        for coh, v in trials_by_cond_wager.items():
            (xbins, ybins, xedges,
             binsizes) = datatools.partition(v['durations'], v['values'], nbins=nbins)
            duration = [np.mean(xbin) for xbin in xbins]
            value    = [np.mean(ybin) for ybin in ybins]
            value_by_duration_by_coh_wager[coh] = (duration, value)

    #=====================================================================================
    # Plot
    #=====================================================================================

    lineprop = {'ls':     '--',
                'lw':     kwargs.get('lw', 1),
                'dashes': kwargs.get('dashes', [9, 4])}
    dataprop = {'mew': kwargs.get('mew', 1)}
    dataprop['ms'] = kwargs.get('ms', 6) + dataprop['mew']/2

    lineprop_wager = {'lw':  kwargs.get('lw', 1)}
    dataprop_wager = {'ms':  kwargs.get('ms', 7),
                      'mew': kwargs.get('mew', 0)}

    colors = kwargs.get('colors', kiani2009_colors)

    # No-wager trials
    cohs = sorted(value_by_duration_by_coh)
    for coh in cohs:
        duration, value = value_by_duration_by_coh[coh]

        plot.plot(duration, value, color=colors[coh], zorder=10, **lineprop)
        plot.plot(duration, value, 'o', mfc='w', mec=colors[coh],
                  zorder=10, **dataprop)

    # Wager trials
    cohs = sorted(value_by_duration_by_coh_wager)
    for coh in cohs:
        duration, value = value_by_duration_by_coh_wager[coh]

        plot.plot(duration, value, color=colors[coh], zorder=5, **lineprop_wager)
        plot.plot(duration, value, 'o', mfc=colors[coh], mec=colors[coh],
                  zorder=5, **dataprop_wager)

    plot.xlim(100, 800)
    #plot.ylim(0.5, 1)

    #=====================================================================================

    return value_by_duration_by_coh, value_by_duration_by_coh_wager

#/////////////////////////////////////////////////////////////////////////////////////////

def compute_dprime(trials, perf, r):
    """
    Compute d' for choice.

    """
    N  = r.shape[-1]
    L  = np.zeros(N)
    L2 = np.zeros(N)
    R  = np.zeros(N)
    R2 = np.zeros(N)
    nL = 0
    nR = 0

    for n, trial in enumerate(trials):
        if perf.choices[n] is None:
            continue

        stimulus = trial['epochs']['stimulus']
        r_n      = r[stimulus,n]

        if trial['left_right'] < 0:
            L  += np.sum(r_n,    axis=0)
            L2 += np.sum(r_n**2, axis=0)
            nL += r_n.shape[0]
        else:
            R  += np.sum(r_n,    axis=0)
            R2 += np.sum(r_n**2, axis=0)
            nR += r_n.shape[0]

    mean_L = L/nL
    var_L  = L2/nL - mean_L**2

    mean_R = R/nR
    var_R  = R2/nR - mean_R**2

    return -utils.div(mean_L - mean_R, np.sqrt((var_L + var_R)/2))

def get_preferred_targets(trials, perf, r):
    """
    Determine preferred targets.

    """
    dprime = compute_dprime(trials, perf, r)
    for i in xrange(len(dprime)):
        if abs(dprime[i]) > 0.5:
            print(i, dprime[i])

    return 2*(dprime > 0) - 1

def sort(trialsfile, plots, unit=None, network='p', **kwargs):
    # Load trials
    data = utils.load(trialsfile)
    if len(data) == 9:
        trials, U, Z, A, P, M, perf, r_p, r_v = data
    else:
        trials, U, Z, Z_b, A, P, M, perf, r_p, r_v = data

    if network == 'p':
        print("Sorting policy network activity.")
        r = r_p
    else:
        print("Sorting value network activity.")
        r = r_v

    # Number of units
    N = r.shape[-1]

    # Time
    time = trials[0]['time']
    Ntime = len(time)

    # Aligned time
    time_a  = np.concatenate((-time[1:][::-1], time))
    Ntime_a = len(time_a)

    #=====================================================================================
    # Preferred targets
    #=====================================================================================

    preferred_targets = get_preferred_targets(trials, perf, r)

    #=====================================================================================
    # No-wager trials
    #=====================================================================================

    def get_no_wager(func_t0):
        trials_by_cond = {}
        for n, trial in enumerate(trials):
            if trial['wager']:
                continue

            if trial['coh'] == 0:
                continue

            if perf.choices[n] is None:
                continue

            cond = trial['left_right']

            m_n = np.tile(M[:,n], (N, 1)).T
            r_n = r[:,n]*m_n

            t0 = func_t0(trial['epochs'], perf.t_choices[n])

            # Storage
            trials_by_cond.setdefault(cond, {'r': np.zeros((Ntime_a, N)),
                                             'n': np.zeros((Ntime_a, N))})

            # Before
            n_b = r_n[:t0].shape[0]
            trials_by_cond[cond]['r'][Ntime-1-n_b:Ntime-1] += r_n[:t0]
            trials_by_cond[cond]['n'][Ntime-1-n_b:Ntime-1] += m_n[:t0]

            # After
            n_a = r_n[t0:].shape[0]
            trials_by_cond[cond]['r'][Ntime-1:Ntime-1+n_a] += r_n[t0:]
            trials_by_cond[cond]['n'][Ntime-1:Ntime-1+n_a] += m_n[t0:]

        # Average
        for cond in trials_by_cond:
            trials_by_cond[cond] = utils.div(trials_by_cond[cond]['r'],
                                             trials_by_cond[cond]['n'])

        return trials_by_cond

    noTs_stimulus = get_no_wager(lambda epochs, t_choice: epochs['stimulus'][0] - 1)
    noTs_choice   = get_no_wager(lambda epochs, t_choice: t_choice)

    #=====================================================================================
    # Wager trials, aligned to stimulus onset
    #=====================================================================================

    def get_wager(func_t0):
        trials_by_cond      = {}
        trials_by_cond_sure = {}
        for n, trial in enumerate(trials):
            if not trial['wager']:
                continue

            if perf.choices[n] is None:
                continue

            if trial['coh'] == 0:
                continue

            cond = trial['left_right']

            m_n = np.tile(M[:,n], (N, 1)).T
            r_n = r[:,n]*m_n

            t0 = func_t0(trial['epochs'], perf.t_choices[n])

            if perf.choices[n] == 'S':
                # Storage
                trials_by_cond_sure.setdefault(cond, {'r': np.zeros((Ntime_a, N)),
                                                      'n': np.zeros((Ntime_a, N))})

                # Before
                n_b = r_n[:t0].shape[0]
                trials_by_cond_sure[cond]['r'][Ntime-1-n_b:Ntime-1] += r_n[:t0]
                trials_by_cond_sure[cond]['n'][Ntime-1-n_b:Ntime-1] += m_n[:t0]

                # After
                n_a = r_n[t0:].shape[0]
                trials_by_cond_sure[cond]['r'][Ntime-1:Ntime-1+n_a] += r_n[t0:]
                trials_by_cond_sure[cond]['n'][Ntime-1:Ntime-1+n_a] += m_n[t0:]
            else:
                # Storage
                trials_by_cond.setdefault(cond, {'r': np.zeros((Ntime_a, N)),
                                                 'n': np.zeros((Ntime_a, N))})

                # Before
                n_b = r_n[:t0].shape[0]
                trials_by_cond[cond]['r'][Ntime-1-n_b:Ntime-1] += r_n[:t0]
                trials_by_cond[cond]['n'][Ntime-1-n_b:Ntime-1] += m_n[:t0]

                # After
                n_a = r_n[t0:].shape[0]
                trials_by_cond[cond]['r'][Ntime-1:Ntime-1+n_a] += r_n[t0:]
                trials_by_cond[cond]['n'][Ntime-1:Ntime-1+n_a] += m_n[t0:]

        # Average
        for cond in trials_by_cond:
            trials_by_cond[cond] = utils.div(trials_by_cond[cond]['r'],
                                             trials_by_cond[cond]['n'])

        # Average
        for cond in trials_by_cond_sure:
            trials_by_cond_sure[cond] = utils.div(trials_by_cond_sure[cond]['r'],
                                                  trials_by_cond_sure[cond]['n'])

        return trials_by_cond, trials_by_cond_sure

    Ts_stimulus, Ts_stimulus_sure = get_wager(lambda epochs, t_choice: epochs['stimulus'][0] - 1)
    Ts_sure, Ts_sure_sure         = get_wager(lambda epochs, t_choice: epochs['sure'][0] - 1)
    Ts_choice, Ts_choice_sure     = get_wager(lambda epochs, t_choice: t_choice)

    #=====================================================================================
    # Plot
    #=====================================================================================

    lw     = kwargs.get('lw', 1.25)
    dashes = kwargs.get('dashes', [3, 1.5])

    in_opp_colors = {-1: '0.6', +1: 'k'}

    def plot_noTs(noTs, plot, unit, tmin, tmax):
        w,   = np.where((tmin <= time_a) & (time_a <= tmax))
        t    = time_a[w]
        yall = [[1]]

        for lr in noTs:
            color = in_opp_colors[lr*preferred_targets[unit]]
            y = noTs[lr][w,unit]
            plot.plot(t, y, color=color, lw=lw)
            yall.append(y)

        plot.xlim(tmin, tmax)
        plot.xticks([0, tmax])
        plot.lim('y', yall, lower=0)

        return yall

    def plot_Ts(Ts, Ts_sure, plot, unit, tmin, tmax):
        w,   = np.where((tmin <= time_a) & (time_a <= tmax))
        t    = time_a[w]
        yall = [[1]]

        for lr in Ts:
            color = in_opp_colors[lr*preferred_targets[unit]]
            y = Ts[lr][w,unit]
            plot.plot(t, y, color=color, lw=lw)
            yall.append(y)
        for lr in Ts_sure:
            color = in_opp_colors[lr*preferred_targets[unit]]
            y = Ts_sure[lr][w,unit]
            plot.plot(t, y, color=color, lw=lw, linestyle='--', dashes=dashes)
            yall.append(y)

        plot.xlim(tmin, tmax)
        plot.xticks([0, tmax])
        plot.lim('y', yall, lower=0)

        return yall

    if unit is not None:
        y = []

        tmin = kwargs.get('noTs-stimulus-tmin', -100)
        tmax = kwargs.get('noTs-stimulus-tmax', 700)
        y += plot_noTs(noTs_stimulus, plots['noTs-stimulus'], unit, tmin, tmax)

        tmin = kwargs.get('noTs-choice-tmin', -500)
        tmax = kwargs.get('noTs-choice-tmax', 0)
        y += plot_noTs(noTs_choice, plots['noTs-choice'], unit, tmin, tmax)

        tmin = kwargs.get('Ts-stimulus-tmin', -100)
        tmax = kwargs.get('Ts-stimulus-tmax', 700)
        y += plot_Ts(Ts_stimulus, Ts_stimulus_sure, plots['Ts-stimulus'], unit, tmin, tmax)

        tmin = kwargs.get('Ts-sure-tmin', -200)
        tmax = kwargs.get('Ts-sure-tmax', 700)
        y += plot_Ts(Ts_sure, Ts_sure_sure, plots['Ts-sure'], unit, tmin, tmax)

        tmin = kwargs.get('Ts-choice-tmin', -500)
        tmax = kwargs.get('Ts-choice-tmax', 0)
        y += plot_Ts(Ts_choice, Ts_choice_sure, plots['Ts-choice'], unit, tmin, tmax)

        return y
    else:
        name = plots
        for unit in xrange(N):
            w   = utils.mm_to_inch(174)
            r   = 0.35
            fig = Figure(w=w, r=r)

            x0 = 0.09
            y0 = 0.15

            w = 0.13
            h = 0.75

            dx = 0.05
            DX = 0.08

            fig.add('noTs-stimulus', [x0, y0, w, h])
            fig.add('noTs-choice',   [fig[-1].right+dx, y0, w, h])
            fig.add('Ts-stimulus',   [fig[-1].right+DX, y0, w, h])
            fig.add('Ts-sure',       [fig[-1].right+dx, y0, w, h])
            fig.add('Ts-choice',     [fig[-1].right+dx, y0, w, h])

            #-----------------------------------------------------------------------------

            y = []

            plot = fig['noTs-stimulus']
            y += plot_noTs(noTs_stimulus, plot, unit, -100, 700)
            plot.vline(0)

            plot = fig['noTs-choice']
            y += plot_noTs(noTs_choice, plot, unit, -500, 200)
            plot.vline(0)

            plot = fig['Ts-stimulus']
            y += plot_Ts(Ts_stimulus, Ts_stimulus_sure, plot, unit, -100, 700)
            plot.vline(0)

            plot = fig['Ts-sure']
            y += plot_Ts(Ts_sure, Ts_sure_sure, plot, unit, -200, 700)
            plot.vline(0)

            plot = fig['Ts-choice']
            y += plot_Ts(Ts_choice, Ts_choice_sure, plot, unit, -500, 200)
            plot.vline(0)

            for plot in fig.plots.values():
                plot.lim('y', y, lower=0)

            #-----------------------------------------------------------------------------

            fig.save(name+'_{}{:03d}'.format(network, unit))
            fig.close()

#/////////////////////////////////////////////////////////////////////////////////////////

def do(action, args, config):
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    #=====================================================================================

    if 'trials' in action:
        try:
            trials_per_condition = int(args[0])
        except:
            trials_per_condition = 100

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'], config['dt'])

        spec         = model.spec
        wagers       = spec.wagers
        left_rights  = spec.left_rights
        cohs         = spec.cohs
        n_conditions = spec.n_conditions
        n_trials     = trials_per_condition * n_conditions

        print("{} trials".format(n_trials))
        task   = model.Task()
        trials = []
        for n in xrange(n_trials):
            k = tasktools.unravel_index(n, (len(wagers), len(left_rights), len(cohs)))
            context = {
                'wager':      wagers[k.pop(0)],
                'left_right': left_rights[k.pop(0)],
                'coh':        cohs[k.pop(0)]
                }
            trials.append(task.get_condition(pg.rng, pg.dt, context))
        runtools.run(action, trials, pg, config['trialspath'])

    #=====================================================================================

    elif action == 'sure_stimulus_duration':
        trialsfile = runtools.behaviorfile(config['trialspath'])

        fig  = Figure()
        plot = fig.add()

        sure_stimulus_duration(trialsfile, plot)

        plot.xlabel('Stimulus duration (ms)')
        plot.ylabel('Probability sure target')

        fig.save(path=config['figspath'], name=action)

    #=====================================================================================

    elif action == 'correct_stimulus_duration':
        trialsfile = runtools.behaviorfile(config['trialspath'])

        fig  = Figure()
        plot = fig.add()

        correct_stimulus_duration(trialsfile, plot)

        plot.xlabel('Stimulus duration (ms)')
        plot.ylabel('Probability correct')

        fig.save(path=config['figspath'], name=action)

    #=====================================================================================

    elif action == 'value_stimulus_duration':
        trialsfile = runtools.activityfile(config['trialspath'])

        fig  = Figure()
        plot = fig.add()

        value_stimulus_duration(trialsfile, plot)

        plot.xlabel('Stimulus duration (ms)')
        plot.ylabel('Expected reward')

        fig.save(path=config['figspath'], name=action)

    #=====================================================================================

    elif action == 'sort':
        if 'value' in args:
            network = 'v'
        else:
            network = 'p'

        trialsfile = runtools.activityfile(config['trialspath'])
        sort(trialsfile, os.path.join(config['figspath'], 'sorted'), network=network)
