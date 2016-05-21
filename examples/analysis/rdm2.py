from __future__ import absolute_import, division

import os

import numpy as np

from pyrl          import datatools, fittools, tasktools, utils
from pyrl.figtools import Figure

#/////////////////////////////////////////////////////////////////////////////////////////

THIS = 'pyrl.analysis.rdm2'

#/////////////////////////////////////////////////////////////////////////////////////////

colors = {
    0:    '#c6dbef',
    3.2:  '#9ecae1',
    6.4:  '#6baed6',
    12.8: '#4292c6',
    25.6: '#2171b5',
    51.2: '#084594'
    }

colors_kiani2009 = {
    0:    '#d53137',
    3.2:  '#9e9c3f',
    6.4:  '#1fa54d',
    12.8: '#2f377c',
    25.6: '#1a1919',
    51.2: '#d52f81'
    }

#/////////////////////////////////////////////////////////////////////////////////////////

def psychometric(trialsfile, m, plot, plot_decision=True, **kwargs):
    # Load trials
    trials, A, R, M, perf = utils.load(trialsfile)

    decision_by_coh = {}
    right_by_coh    = {}
    for n, trial in enumerate(trials):
        coh = trial['left_right']*trial['coh']
        decision_by_coh.setdefault(coh, [])
        if perf.decisions[n]:
            decision_by_coh[coh].append(1)

            t = int(np.sum(M[:,n]))
            if A[t-1,n,m.actions['SACCADE_RIGHT']] == 1:
                right = 1
            else:
                right = 0

            right_by_coh.setdefault(coh, []).append(right)
        else:
            decision_by_coh[coh].append(0)

    cohs       = np.sort(right_by_coh.keys())
    p_decision = np.zeros(len(cohs))
    p_right    = np.zeros(len(cohs))
    for i, coh in enumerate(cohs):
        p_decision[i] = sum(decision_by_coh[coh])/len(decision_by_coh[coh])
        p_right[i]    = utils.divide(sum(right_by_coh[coh]), len(right_by_coh[coh]))

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    # Plot properties
    ms = kwargs.get('ms', 6)
    lw = kwargs.get('lw', 1.5)

    # Prop. decision
    if plot_decision:
        plot.plot(cohs, 100*p_decision, lw=lw, color=Figure.colors('orange'),
                  label='$P$(decision)')
        plot.plot(cohs, 100*p_decision, 'o', ms=ms, mew=0, mfc=Figure.colors('orange'))

    # Fit psychometric curve
    props = dict(lw=lw, color=Figure.colors('blue'), label='$P$(right$|$decision)')
    try:
        popt, func = fittools.fit_psychometric(cohs, p_right)

        fit_cohs = np.linspace(min(cohs), max(cohs), 201)
        fit_pr   = func(fit_cohs, **popt)
        plot.plot(fit_cohs, 100*fit_pr, **props)
    except RuntimeError:
        print("Unable to fit, drawing a line through the points.")
        plot.plot(cohs, 100*p_right, **props)
    plot.plot(cohs, 100*p_right, 'o', ms=ms, mew=0, mfc=Figure.colors('blue'))

    plot.xlim(cohs[0], cohs[-1])
    plot.ylim(0, 100)

    #plot.hline(50, linestyle='--')

    plot.xlabel('Percent coherence toward right')
    plot.ylabel('Percent decision/right')

    if kwargs.get('legend', False):
        props = {'prop': {'size': 8.5}, 'handlelength': 1.4,
                 'handletextpad': 1.2, 'labelspacing': 0.9}
        plot.legend(bbox_to_anchor=(0.97, 0.17), **props)

#/////////////////////////////////////////////////////////////////////////////////////////

def chronometric(trialsfile, plot, **kwargs):
    """
    Reaction time as a function of coherence.

    """
    # Load trials
    trials, A, R, M, perf = utils.load(trialsfile)

    zero_coherence_rt = []
    correct_rt_by_coh = {}
    error_rt_by_coh   = {}
    no_decision       = 0
    for n, trial in enumerate(trials):
        coh = trial['coh']
        if coh == 0:
            if perf.decisions[n]:
                stimulus_start = trial['durations']['stimulus'][0]
                rt = trial['time'][np.sum(M[:,n])-1] - stimulus_start
                zero_coherence_rt.append(rt)
            continue

        if perf.decisions[n]:
            stimulus_start = trial['durations']['stimulus'][0]
            rt = trial['time'][np.sum(M[:,n])-1] - stimulus_start
            if perf.corrects[n]:
                correct_rt_by_coh.setdefault(coh,[]).append(rt)
            else:
                error_rt_by_coh.setdefault(coh, []).append(rt)
        else:
            no_decision += 1

    min_trials = 0

    # Correct trials
    print("Correct trials.")
    correct_cohs = np.sort(correct_rt_by_coh.keys())
    correct_rt   = np.zeros(len(correct_cohs))
    correct_idx  = []
    correct_tot  = 0
    correct_n    = 0
    for i, coh in enumerate(correct_cohs):
        if len(correct_rt_by_coh[coh]) > 0:
            correct_tot += np.sum(correct_rt_by_coh[coh])
            correct_n   += len(correct_rt_by_coh[coh])
            correct_rt[i] = np.mean(correct_rt_by_coh[coh])
            print(coh, len(correct_rt_by_coh[coh]), correct_rt[i], np.min(correct_rt_by_coh[coh]), np.max(correct_rt_by_coh[coh]))
            if len(correct_rt_by_coh[coh]) > min_trials:
                correct_idx.append(i)

    # Error trials
    print("Error trials.")
    error_cohs = np.sort(error_rt_by_coh.keys())
    error_rt   = np.zeros(len(error_cohs))
    error_idx  = []
    error_tot  = 0
    error_n    = 0
    for i, coh in enumerate(error_cohs):
        if len(error_rt_by_coh[coh]) > 0:
            error_tot += np.sum(error_rt_by_coh[coh])
            error_n   += len(error_rt_by_coh[coh])
            error_rt[i] = np.mean(error_rt_by_coh[coh])
            print(coh, len(error_rt_by_coh[coh]), error_rt[i], np.min(error_rt_by_coh[coh]), np.max(error_rt_by_coh[coh]))
            if len(error_rt_by_coh[coh]) > min_trials:
                error_idx.append(i)

    print("[ {}.chronometric ]".format(THIS))
    print("  {}/{} non-decision trials.".format(no_decision, len(trials)))
    print("  Mean RT, correct trials: {:.2f} ms".format(correct_tot/correct_n))
    print("  Mean RT, error trials:   {:.2f} ms".format(error_tot/error_n))

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    # Correct trials
    prop = {'color': kwargs.get('color', '0.2'),
            'lw':    kwargs.get('lw', 1)}
    plot.plot(correct_cohs[correct_idx], correct_rt[correct_idx], **prop)
    prop = {'marker':    'o',
            'linestyle': 'none',
            'ms':        kwargs.get('ms',  6),
            'mfc':       '0.2',
            'mew':       0}
    plot.plot(correct_cohs[correct_idx], correct_rt[correct_idx], **prop)

    # Error trials
    prop = {'color': kwargs.get('color', '0.2'),
            'lw':    kwargs.get('lw', 1)}
    plot.plot(error_cohs[error_idx], error_rt[error_idx], **prop)
    prop = {'marker':    'o',
            'linestyle': 'none',
            'ms':        kwargs.get('ms',  6) - 1,
            'mfc':       'w',
            'mec':       '0.2',
            'mew':       kwargs.get('mew', 1)}
    plot.plot(error_cohs[error_idx], error_rt[error_idx], **prop)

    #plot.lim('y', [correct_rt, error_rt], lower=0)
    plot.ylim(200, 1000)

    plot.xscale('log')
    plot.xticks([1, 10, 100])
    plot.xticklabels([1, 10, 100])

def sort_rt(trialsfile, figspath, name, plot=True, **kwargs):
    """
    Sort reaction-time trials.

    """
    # Load trials
    trials, U, Q, Z, A, R, M, perf, states, baseline_states = utils.load(trialsfile)

    # Data shape
    Ntime = states.shape[0]
    N     = states.shape[-1]

    #-------------------------------------------------------------------------------------
    # Aligned to motion onset
    #-------------------------------------------------------------------------------------

    states_by_coh_motion   = {}
    n_states_by_coh_motion = {}
    for n, trial in enumerate(trials):
        if not perf.corrects[n] and trial['coh'] != 0:
            continue

        lr  = trial['left_right']
        coh = lr*trial['coh']

        states_by_coh_motion.setdefault(coh, np.zeros((Ntime, N)))
        n_states_by_coh_motion.setdefault(coh, np.zeros((Ntime, N)))

        t = np.sum(M[:,n])
        Mn = np.tile(M[:,n], (N,1)).T

        states_by_coh_motion[coh][:] += states[:,n]*Mn
        n_states_by_coh_motion[coh][:] += Mn

    for coh in states_by_coh_motion:
        for t in xrange(Ntime):
            states_by_coh_motion[coh][t] = utils.div(states_by_coh_motion[coh][t],
                                                     n_states_by_coh_motion[coh][t])

    #-------------------------------------------------------------------------------------
    # Aligned to saccade
    #-------------------------------------------------------------------------------------

    states_by_coh_saccade   = {}
    n_states_by_coh_saccade = {}
    for n, trial in enumerate(trials):
        if not perf.corrects[n] and trial['coh'] != 0:
            continue

        lr  = trial['left_right']
        coh = lr*trial['coh']

        states_by_coh_saccade.setdefault(coh, np.zeros((Ntime, N)))
        n_states_by_coh_saccade.setdefault(coh, np.zeros((Ntime, N)))

        t = np.sum(M[:,n])
        Mn = np.tile(M[:t,n], (N,1)).T

        states_by_coh_saccade[coh][-t:] += states[:t,n]*Mn
        n_states_by_coh_saccade[coh][-t:] += Mn

    for coh in states_by_coh_saccade:
        for t in xrange(Ntime):
            states_by_coh_saccade[coh][t] = utils.div(states_by_coh_saccade[coh][t],
                                                      n_states_by_coh_saccade[coh][t])

    #-------------------------------------------------------------------------------------

    if not plot:
        return None

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    lw = kwargs.get('lw', 1.5)

    time = trials[0]['time']

    lr_colors = {1: '0', -1: '0.5'}

    for unit in xrange(N):
        w = 174/25.4
        r = 0.5
        h = r*w
        fig = Figure(w=w, h=h)

        x0 = 0.07
        y0 = 0.1
        w  = 0.37
        h  = 0.8
        dx = 1.3*w
        plots = {
            'motion':  fig.add([x0,    y0, w, h]),
            'saccade': fig.add([x0+dx, y0, w, h])
            }

        #---------------------------------------------------------------------------------
        # Aligned to motion onset
        #---------------------------------------------------------------------------------

        plot = plots['motion']

        motion_onset = trials[0]['durations']['stimulus'][0]
        t_aligned = time[:-1] - motion_onset
        for coh in sorted(states_by_coh_motion.keys()):
            if coh > 0:
                ls = '-'
            else:
                ls = '--'
            plot.plot(t_aligned, states_by_coh_motion[coh][:,unit],
                      ls=ls, lw=lw, color=colors[abs(coh)], label='{}\%'.format(coh))

        plot.xlim(t_aligned[0], t_aligned[-1])
        plot.ylim(0, 1)

        #---------------------------------------------------------------------------------
        # Aligned to saccade
        #---------------------------------------------------------------------------------

        plot = plots['saccade']

        t_aligned = time[:-1] - max(time[:-1])
        for coh in sorted(states_by_coh_saccade.keys()):
            if coh > 0:
                ls = '-'
            else:
                ls = '--'
            plot.plot(t_aligned, states_by_coh_saccade[coh][:,unit],
                      ls=ls, lw=lw, color=colors[abs(coh)], label='{}\%'.format(coh))

        plot.xlim(t_aligned[0], t_aligned[-1])
        plot.ylim(0, 1)

        #---------------------------------------------------------------------------------

        fig.save(path=figspath, name=name+'_{:03d}'.format(unit))
        fig.close()

#/////////////////////////////////////////////////////////////////////////////////////////

def sure_stimulus_duration(trialsfile, plot, saved=None, **kwargs):
    """
    Probability sure target as a function of stimulus duration.

    """
    if saved is None:
        # Load trials
        trials, A, R, M, perf = utils.load(trialsfile)

        sure_duration_by_coh = {}
        for n, trial in enumerate(trials):
            if not trial['wager']:
                continue

            coh = trial['coh']
            if perf.sure_decisions[n]:
                stimulus = np.ptp(trial['durations']['stimulus'])
                sure_duration_by_coh.setdefault(coh, ([], []))[0].append(perf.sures[n])
                sure_duration_by_coh[coh][1].append(stimulus)

        nbins = kwargs.get('nbins', 5)

        sure_by_coh = {}
        for coh, (sure, duration) in sure_duration_by_coh.items():
            Xbins, Ybins, Xedges, _ = datatools.partition(np.asarray(duration),
                                                          np.asarray(sure),
                                                          nbins=nbins)
            sure_by_coh[coh] = ([np.mean(Xbin) for Xbin in Xbins],
                                [utils.divide(np.sum(Ybin > 0),
                                 len(Ybin)) for Ybin in Ybins])
    else:
        sure_by_coh = saved

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    lineprop = {'lw':  kwargs.get('lw', 1)}
    dataprop = {'ms':  kwargs.get('ms', 6),
                'mew': kwargs.get('mew', 0)}

    # Colors
    colors = colors_kiani2009

    cohs = sorted(sure_by_coh)
    xall = []
    for coh in cohs:
        stim, sure = sure_by_coh[coh]

        plot.plot(stim, sure, color=colors[coh], label='{}\%'.format(coh),
                  **lineprop)
        plot.plot(stim, sure, 'o', mfc=colors[coh], **dataprop)
        xall.append(stim)

    #plot.lim('x', xall)
    plot.xlim(100, 800)
    plot.ylim(0, 1)

    plot.xlabel('Stimulus duration (ms)')
    plot.ylabel('Probability sure target')

    #-------------------------------------------------------------------------------------

    return sure_by_coh

#/////////////////////////////////////////////////////////////////////////////////////////

def stimulus_duration(trialsfile, plot, wager=False, saved=None, **kwargs):
    """
    Percent correct as a function of stimulus duration.

    """
    if saved is None:
        # Load trials
        trials, A, R, M, perf = utils.load(trialsfile)

        correct_duration_by_coh        = {}
        correct_duration_by_coh_waived = {}
        for n, trial in enumerate(trials):
            coh = trial['coh']
            if coh == 0:
                continue

            if perf.decisions[n]:
                if 'durations' in trial:
                    stimulus = np.ptp(trial['durations']['stimulus'])
                else:
                    stimulus = np.ptp(trial['epoch_durations']['stimulus'])
                if trial.get('wager', False):
                    correct_duration_by_coh_waived.setdefault(coh, ([], []))[0].append(perf.corrects[n])
                    correct_duration_by_coh_waived[coh][1].append(stimulus)
                else:
                    correct_duration_by_coh.setdefault(coh, ([], []))[0].append(perf.corrects[n])
                    correct_duration_by_coh[coh][1].append(stimulus)

        # Number of bins
        nbins = kwargs.get('nbins', 5)

        # No-wager trials
        correct_by_coh = {}
        for coh, (correct, duration) in correct_duration_by_coh.items():
            Xbins, Ybins, Xedges, _ = datatools.partition(np.asarray(duration), np.asarray(correct),
                                                nbins=nbins)
            correct_by_coh[coh] = ([np.mean(Xbin) for Xbin in Xbins],
                                   [utils.divide(np.sum(Ybin > 0), len(Ybin))
                                    for Ybin in Ybins])

        # Sure bet presented but waived
        correct_by_coh_waived = {}
        if correct_duration_by_coh_waived:
            correct_by_coh_waived = {}
            for coh, (correct, duration) in correct_duration_by_coh_waived.items():
                Xbins, Ybins, Xedges, _ = datatools.partition(np.asarray(duration), np.asarray(correct),
                                                    nbins=nbins)
                correct_by_coh_waived[coh] = ([np.mean(Xbin) for Xbin in Xbins],
                                              [utils.divide(np.sum(Ybin > 0), len(Ybin))
                                               for Ybin in Ybins])
    else:
        correct_by_coh, correct_by_coh_waived = saved

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    if wager:
        lineprop = {'ls': '--',
                    'lw': kwargs.get('lw', 1.5),
                    'dashes': kwargs.get('dashes', [10, 5])}
        dataprop = {'ms':     kwargs.get('ms', 7),
                    'mew':    kwargs.get('mew', 1)}
        dataprop['ms'] += dataprop['mew']/2

        lineprop_waived = {'ls': '-',
                           'lw': kwargs.get('lw', 1.5)}
        dataprop_waived = {'ms':  kwargs.get('ms', 7),
                           'mew': kwargs.get('mew', 0)}

        # To determine x-limits
        xall = []

        # No-wager trials
        cohs = sorted(correct_by_coh)
        for coh in cohs:
            stim, correct = correct_by_coh[coh]

            plot.plot(stim, correct, color=colors_kiani2009[coh], zorder=10, **lineprop)
            plot.plot(stim, correct, 'o', mfc='w', mec=colors_kiani2009[coh], zorder=10, **dataprop)
            xall.append(stim)

        # Sure bet presented but waived
        if correct_by_coh_waived:
            cohs = sorted(correct_by_coh_waived)
            for coh in cohs:
                stim, correct = correct_by_coh_waived[coh]

                plot.plot(stim, correct, color=colors_kiani2009[coh], label='{}\%'.format(coh),
                          zorder=5, **lineprop_waived)
                plot.plot(stim, correct, 'o', mfc=colors_kiani2009[coh],
                          mec=colors_kiani2009[coh], zorder=5, **dataprop_waived)
                xall.append(stim)

        rvals = [correct_by_coh, correct_by_coh_waived]
    else:
        lineprop = {'ls': '-', 'lw': kwargs.get('lw', 1)}
        dataprop = {'ms':  kwargs.get('ms', 6),
                    'mew': kwargs.get('mew', 0)}

        # To determine x-limits
        xall = []

        if correct_duration_by_coh:
            def fit_func(t, inv_tau, p_inf):
                return 0.5 + (1 - np.exp(-inv_tau*t))*(p_inf - 0.5)
            fitparams = [0.1, 0.75]
            fitbounds = [(0, np.inf), (0.5, 1)]

            '''
            def sigmoid(x):
                return 1/(1 + np.exp(-x))

            def fit_func(t, a, b):
                return a/2 + (1 - a)*sigmoid(b*t)
            fitparams = [0.05, 1/100]
            fitbounds = [(0, 1), (0, np.inf)]
            '''

            # Fits
            fits = {}
            '''
            for coh, (correct, duration) in correct_duration_by_coh.items():
                duration  = np.asarray(duration)
                correct   = 1.*np.asarray(correct)
                try:
                    fits[coh] = fittools.binregress(duration, correct, fit_func,
                                                    fitparams, bounds=fitbounds)
                except:
                    print("Couldn't fit coh = {}%".format(coh))
            print(fits)
            fits = {}
            '''

            cohs = sorted(correct_by_coh)
            for coh in cohs:
                duration, correct = correct_by_coh[coh]

                if coh in fits:
                    fit_duration = np.linspace(min(duration), max(duration), 201)
                    fit_correct  = fit_func(fit_duration, *fits[coh])
                    plot.plot(fit_duration, fit_correct, color=colors_kiani2009[coh], label='{}\%'.format(coh),
                              **lineprop)
                else:
                    plot.plot(duration, correct, color=colors_kiani2009[coh], label='{}\%'.format(coh),
                              **lineprop)
                plot.plot(duration, correct, 'o', mfc=colors_kiani2009[coh], **dataprop)
                xall.append(duration)

        rvals = []

    plot.lim('x', xall)
    plot.ylim(0.5, 1)

    #-------------------------------------------------------------------------------------

    return rvals

#/////////////////////////////////////////////////////////////////////////////////////////

def get_choice_selectivity(trials, actions, A, M, perf, states):
    """
    Compute d' for choice.

    """
    N  = states.shape[-1]
    L  = np.zeros(N)
    L2 = np.zeros(N)
    R  = np.zeros(N)
    R2 = np.zeros(N)
    nL = 0
    nR = 0

    for n, trial in enumerate(trials):
        if trial['coh'] == 0:
            continue

        stimulus = [s for s in trial['epochs']['stimulus']]
        t = int(np.sum(M[:,n])) - 1
        if t >= stimulus[-1]:
            r = np.sum(states[stimulus,n], axis=0)
            if A[t,n,actions['SACCADE_LEFT']] == 1:
                L  += r
                L2 += r**2
                nL += len(stimulus)
            else:
                R  += r
                R2 += r**2
                nR += len(stimulus)

    mean_L = L/nL
    var_L  = L2/nL - mean_L**2

    mean_R = R/nR
    var_R  = R2/nR - mean_R**2

    return utils.div(mean_L - mean_R, np.sqrt((var_L + var_R)/2))

def get_preferred_targets(trials, actions, A, M, perf, states):
    """
    Determine preferred targets.

    """
    dprime = get_choice_selectivity(trials, actions, A, M, perf, states)

    return 2*(dprime < 0) - 1

def sort_postdecision(trialsfile, m, plots, unit=None, **kwargs):
    """
    Sort postdecision trials by condition.

    """
    # Load trials
    trials, U, Q, Z, A, R, M, perf, states, baseline_states = utils.load(trialsfile)

    # Data shape
    Ntime = states.shape[0]
    N     = states.shape[-1]

    # Same for every trial
    time = trials[0]['time']

    # Aligned time
    time_aligned = np.concatenate((-time[1:][::-1], time))

    #=====================================================================================
    # Get the preference for each unit
    #=====================================================================================

    preferred_targets = get_preferred_targets(trials, m.actions, A, M, perf, states)
    for i in xrange(len(preferred_targets)):
        print(i, preferred_targets[i])

    #=====================================================================================
    # Non-wager trials, aligned to dots onset
    #=====================================================================================

    noTs_states_by_lr_dots   = {}
    noTs_n_states_by_lr_dots = {}

    # Sort
    for n, trial in enumerate(trials):
        if trial['wager']:
            continue

        if (trial['coh'] == 0 and perf.decisions[n]) or perf.corrects[n]:
            lr = trial['left_right']

            # Make room
            noTs_states_by_lr_dots.setdefault(lr, np.zeros((2*Ntime-1, N)))
            noTs_n_states_by_lr_dots.setdefault(lr, np.zeros((2*Ntime-1, N), dtype=int))

            # States
            Mn = np.tile(M[:,n], (N,1)).T
            Sn = states[:,n]*Mn

            # Align point
            t0 = trial['epochs']['stimulus'][0] - 1

            #print(Sn[0])
            assert np.all(Sn >= 0), Sn[np.where(Sn < 0)]
            #assert np.all(Sn <= 1), Sn[np.where(Sn > 1)]

            # Before
            n_b = Sn[:t0].shape[0]
            noTs_states_by_lr_dots[lr][Ntime-1-n_b:Ntime-1]   += Sn[:t0]
            noTs_n_states_by_lr_dots[lr][Ntime-1-n_b:Ntime-1] += Mn[:t0]

            # After
            n_f = Sn[t0:].shape[0]
            noTs_states_by_lr_dots[lr][Ntime-1:Ntime-1+n_f]   += Sn[t0:]
            noTs_n_states_by_lr_dots[lr][Ntime-1:Ntime-1+n_f] += Mn[t0:]

    # Average
    for lr in noTs_states_by_lr_dots:
        noTs_states_by_lr_dots[lr] = utils.div(noTs_states_by_lr_dots[lr],
                                               noTs_n_states_by_lr_dots[lr])

    #=====================================================================================
    # Non-wager trials, aligned to saccade
    #=====================================================================================

    noTs_states_by_lr_saccade   = {}
    noTs_n_states_by_lr_saccade = {}

    # Sort
    for n, trial in enumerate(trials):
        if trial['wager']:
            continue

        if (trial['coh'] == 0 and perf.decisions[n]) or perf.corrects[n]:
            lr = trial['left_right']

            # Make room
            noTs_states_by_lr_saccade.setdefault(lr, np.zeros((2*Ntime-1, N)))
            noTs_n_states_by_lr_saccade.setdefault(lr, np.zeros((2*Ntime-1, N), dtype=int))

            # States
            Mn = np.tile(M[:,n], (N,1)).T
            Sn = states[:,n]*Mn

            # t = 0
            t0 = int(np.sum(M[:,n])) - 1

            # Before
            n_b = Sn[:t0].shape[0]
            noTs_states_by_lr_saccade[lr][Ntime-1-n_b:Ntime-1]   += Sn[:t0]
            noTs_n_states_by_lr_saccade[lr][Ntime-1-n_b:Ntime-1] += Mn[:t0]

            # After
            n_f = Sn[t0:].shape[0]
            noTs_states_by_lr_saccade[lr][Ntime-1:Ntime-1+n_f]   += Sn[t0:]
            noTs_n_states_by_lr_saccade[lr][Ntime-1:Ntime-1+n_f] += Mn[t0:]

    # Average
    for lr in noTs_states_by_lr_saccade:
        noTs_states_by_lr_saccade[lr] = utils.div(noTs_states_by_lr_saccade[lr],
                                                  noTs_n_states_by_lr_saccade[lr])

    #=====================================================================================
    # Wager trials, aligned to dots onset
    #=====================================================================================

    Ts_states_by_lr_dots   = {}
    Ts_n_states_by_lr_dots = {}

    Ts_states_by_lr_sure_dots   = {}
    Ts_n_states_by_lr_sure_dots = {}

    # Sort
    for n, trial in enumerate(trials):
        if not trial['wager']:
            continue

        if (trial['coh'] == 0 and perf.sure_decisions[n]) or perf.corrects[n]:
            lr = trial['left_right']

            # States
            Mn = np.tile(M[:,n], (N,1)).T
            Sn = states[:,n]*Mn

            # Align point
            t0 = trial['epochs']['stimulus'][0] - 1

            if not perf.sures[n]:
                # Make room
                Ts_states_by_lr_dots.setdefault(lr, np.zeros((2*Ntime-1, N)))
                Ts_n_states_by_lr_dots.setdefault(lr, np.zeros((2*Ntime-1, N), dtype=int))

                # Before
                n_b = Sn[:t0].shape[0]
                Ts_states_by_lr_dots[lr][Ntime-1-n_b:Ntime-1]   += Sn[:t0]
                Ts_n_states_by_lr_dots[lr][Ntime-1-n_b:Ntime-1] += Mn[:t0]

                # After
                n_f = Sn[t0:].shape[0]
                Ts_states_by_lr_dots[lr][Ntime-1:Ntime-1+n_f]   += Sn[t0:]
                Ts_n_states_by_lr_dots[lr][Ntime-1:Ntime-1+n_f] += Mn[t0:]
            else:
                # Make room
                Ts_states_by_lr_sure_dots.setdefault(lr, np.zeros((2*Ntime-1, N)))
                Ts_n_states_by_lr_sure_dots.setdefault(lr, np.zeros((2*Ntime-1, N), dtype=int))

                # Before
                n_b = Sn[:t0].shape[0]
                Ts_states_by_lr_sure_dots[lr][Ntime-1-n_b:Ntime-1]   += Sn[:t0]
                Ts_n_states_by_lr_sure_dots[lr][Ntime-1-n_b:Ntime-1] += Mn[:t0]

                # After
                n_f = Sn[t0:].shape[0]
                Ts_states_by_lr_sure_dots[lr][Ntime-1:Ntime-1+n_f]   += Sn[t0:]
                Ts_n_states_by_lr_sure_dots[lr][Ntime-1:Ntime-1+n_f] += Mn[t0:]

    # Average
    for lr in Ts_states_by_lr_dots:
        Ts_states_by_lr_dots[lr] = utils.div(Ts_states_by_lr_dots[lr],
                                             Ts_n_states_by_lr_dots[lr])
    for lr in Ts_states_by_lr_sure_dots:
        Ts_states_by_lr_sure_dots[lr] = utils.div(Ts_states_by_lr_sure_dots[lr],
                                                  Ts_n_states_by_lr_sure_dots[lr])

    #=====================================================================================
    # Wager trials, aligned to sure target
    #=====================================================================================

    Ts_states_by_lr_sure   = {}
    Ts_n_states_by_lr_sure = {}

    Ts_states_by_lr_sure_sure   = {}
    Ts_n_states_by_lr_sure_sure = {}

    # Sort
    for n, trial in enumerate(trials):
        if not trial['wager']:
            continue

        if (trial['coh'] == 0 and perf.sure_decisions[n]) or perf.corrects[n]:
            lr = trial['left_right']

            # States
            Mn = np.tile(M[:,n], (N,1)).T
            Sn = states[:,n]*Mn

            # Align point
            t0 = trial['epochs']['sure'][0] - 1

            if not perf.sures[n]:
                # Make room
                Ts_states_by_lr_sure.setdefault(lr, np.zeros((2*Ntime-1, N)))
                Ts_n_states_by_lr_sure.setdefault(lr, np.zeros((2*Ntime-1, N), dtype=int))

                # Before
                n_b = Sn[:t0].shape[0]
                Ts_states_by_lr_sure[lr][Ntime-1-n_b:Ntime-1]   += Sn[:t0]
                Ts_n_states_by_lr_sure[lr][Ntime-1-n_b:Ntime-1] += Mn[:t0]

                # After
                n_f = Sn[t0:].shape[0]
                Ts_states_by_lr_sure[lr][Ntime-1:Ntime-1+n_f]   += Sn[t0:]
                Ts_n_states_by_lr_sure[lr][Ntime-1:Ntime-1+n_f] += Mn[t0:]
            else:
                # Make room
                Ts_states_by_lr_sure_sure.setdefault(lr, np.zeros((2*Ntime-1, N)))
                Ts_n_states_by_lr_sure_sure.setdefault(lr, np.zeros((2*Ntime-1, N), dtype=int))

                # Before
                n_b = Sn[:t0].shape[0]
                Ts_states_by_lr_sure_sure[lr][Ntime-1-n_b:Ntime-1]   += Sn[:t0]
                Ts_n_states_by_lr_sure_sure[lr][Ntime-1-n_b:Ntime-1] += Mn[:t0]

                # After
                n_f = Sn[t0:].shape[0]
                Ts_states_by_lr_sure_sure[lr][Ntime-1:Ntime-1+n_f]   += Sn[t0:]
                Ts_n_states_by_lr_sure_sure[lr][Ntime-1:Ntime-1+n_f] += Mn[t0:]

    # Average
    for lr in Ts_states_by_lr_sure:
        Ts_states_by_lr_sure[lr] = utils.div(Ts_states_by_lr_sure[lr],
                                             Ts_n_states_by_lr_sure[lr])
    for lr in Ts_states_by_lr_sure_sure:
        Ts_states_by_lr_sure_sure[lr] = utils.div(Ts_states_by_lr_sure_sure[lr],
                                                  Ts_n_states_by_lr_sure_sure[lr])

    #=====================================================================================
    # Wager trials, aligned to saccade
    #=====================================================================================

    Ts_states_by_lr_saccade   = {}
    Ts_n_states_by_lr_saccade = {}

    Ts_states_by_lr_sure_saccade   = {}
    Ts_n_states_by_lr_sure_saccade = {}

    # Sort
    for n, trial in enumerate(trials):
        if not trial['wager']:
            continue

        if (trial['coh'] == 0 and perf.sure_decisions[n]) or perf.corrects[n]:
            lr = trial['left_right']

            # States
            Mn = np.tile(M[:,n], (N,1)).T
            Sn = states[:,n]*Mn

            # t = 0
            t0 = int(np.sum(M[:,n])) - 1

            if not perf.sures[n]:
                # Make room
                Ts_states_by_lr_saccade.setdefault(lr, np.zeros((2*Ntime-1, N)))
                Ts_n_states_by_lr_saccade.setdefault(lr, np.zeros((2*Ntime-1, N), dtype=int))

                # Before
                n_b = Sn[:t0].shape[0]
                Ts_states_by_lr_saccade[lr][Ntime-1-n_b:Ntime-1]   += Sn[:t0]
                Ts_n_states_by_lr_saccade[lr][Ntime-1-n_b:Ntime-1] += Mn[:t0]

                # After
                n_f = Sn[t0:].shape[0]
                Ts_states_by_lr_saccade[lr][Ntime-1:Ntime-1+n_f]   += Sn[t0:]
                Ts_n_states_by_lr_saccade[lr][Ntime-1:Ntime-1+n_f] += Mn[t0:]
            else:
                # Make room
                Ts_states_by_lr_sure_saccade.setdefault(lr, np.zeros((2*Ntime-1, N)))
                Ts_n_states_by_lr_sure_saccade.setdefault(lr, np.zeros((2*Ntime-1, N), dtype=int))

                # Before
                n_b = Sn[:t0].shape[0]
                Ts_states_by_lr_sure_saccade[lr][Ntime-1-n_b:Ntime-1]   += Sn[:t0]
                Ts_n_states_by_lr_sure_saccade[lr][Ntime-1-n_b:Ntime-1] += Mn[:t0]

                # After
                n_f = Sn[t0:].shape[0]
                Ts_states_by_lr_sure_saccade[lr][Ntime-1:Ntime-1+n_f]   += Sn[t0:]
                Ts_n_states_by_lr_sure_saccade[lr][Ntime-1:Ntime-1+n_f] += Mn[t0:]

    # Average
    for lr in Ts_states_by_lr_saccade:
        Ts_states_by_lr_saccade[lr] = utils.div(Ts_states_by_lr_saccade[lr],
                                                Ts_n_states_by_lr_saccade[lr])
    for lr in Ts_states_by_lr_sure_saccade:
        Ts_states_by_lr_sure_saccade[lr] = utils.div(Ts_states_by_lr_sure_saccade[lr],
                                                     Ts_n_states_by_lr_sure_saccade[lr])

    #=====================================================================================
    # Plot functions
    #=====================================================================================

    lw     = kwargs.get('lw', 1.25)
    dashes = kwargs.get('dashes', [3, 1.5])

    vline_props = {'lw': kwargs.get('lw_vline', 0.5)}
    if 'dashes_vline' in kwargs:
        vline_props['linestyle'] = '--'
        vline_props['dashes']    = dashes

    lr_colors = {-1: '0.6', +1: 'k'}
    lineprops = dict(lw=lw)

    def noTs_dots(plot, unit):
        w = np.where((time_aligned >= -100) & (time_aligned <= 700))[0]
        all_y = [[1]]
        for lr in noTs_states_by_lr_dots:
            in_opp = lr*preferred_targets[unit]
            plot.plot(time_aligned[w], noTs_states_by_lr_dots[lr][w,unit],
                      color=lr_colors[in_opp], **lineprops)
            all_y.append(noTs_states_by_lr_dots[lr][w,unit])

        plot.xticks([0, 300, 600])

        plot.xlim(-100, 700)
        #plot.ylim(0, 1)
        plot.lim('y', all_y, lower=0)

        plot.vline(0, **vline_props)

    def noTs_saccade(plot, unit):
        w = np.where((time_aligned >= -500) & (time_aligned <= 200))[0]
        all_y = [[1]]
        for lr in noTs_states_by_lr_saccade:
            in_opp = lr*preferred_targets[unit]
            plot.plot(time_aligned[w], noTs_states_by_lr_saccade[lr][w,unit],
                      color=lr_colors[in_opp], **lineprops)
            all_y.append(noTs_states_by_lr_saccade[lr][w,unit])

        plot.xticks([-300, 0])

        plot.xlim(-500, 200)
        #plot.ylim(0, 1)
        plot.lim('y', all_y, lower=0)

        plot.vline(0, **vline_props)

    def Ts_dots(plot, unit):
        w = np.where((time_aligned >= -100) & (time_aligned <= 700))[0]
        all_y = [[1]]
        for lr in Ts_states_by_lr_sure_dots:
            in_opp = lr*preferred_targets[unit]
            plot.plot(time_aligned[w], Ts_states_by_lr_sure_dots[lr][w,unit],
                      color=lr_colors[in_opp],
                      linestyle='--', dashes=dashes,
                      **lineprops)
            all_y.append(Ts_states_by_lr_sure_dots[lr][w,unit])
        for lr in Ts_states_by_lr_dots:
            in_opp = lr*preferred_targets[unit]
            plot.plot(time_aligned[w], Ts_states_by_lr_dots[lr][w,unit],
                      color=lr_colors[in_opp], **lineprops)
            all_y.append(Ts_states_by_lr_dots[lr][w,unit])

        plot.xticks([0, 300, 600])

        plot.xlim(-100, 700)
        #plot.ylim(0, 1)
        plot.lim('y', all_y, lower=0)

        plot.vline(0, **vline_props)

    def Ts_sure(plot, unit):
        w = np.where((time_aligned >= -200) & (time_aligned <= 700))[0]
        all_y = [[1]]
        for lr in Ts_states_by_lr_sure_sure:
            in_opp = lr*preferred_targets[unit]
            plot.plot(time_aligned[w], Ts_states_by_lr_sure_sure[lr][w,unit],
                      color=lr_colors[in_opp],
                      linestyle='--', dashes=dashes,
                      **lineprops)
            all_y.append(Ts_states_by_lr_sure_sure[lr][w,unit])
        for lr in Ts_states_by_lr_sure:
            in_opp = lr*preferred_targets[unit]
            plot.plot(time_aligned[w], Ts_states_by_lr_sure[lr][w,unit],
                      color=lr_colors[in_opp], **lineprops)
            all_y.append(Ts_states_by_lr_dots[lr][w,unit])

        plot.xticks([0, 300, 600])

        plot.xlim(-200, 700)
        #plot.ylim(0, 1)
        plot.lim('y', all_y, lower=0)

        plot.vline(0, **vline_props)

    def Ts_saccade(plot, unit):
        w = np.where((time_aligned >= -500) & (time_aligned <= 200))[0]
        all_y = [[1]]
        for lr in Ts_states_by_lr_sure_saccade:
            in_opp = lr*preferred_targets[unit]
            plot.plot(time_aligned[w], Ts_states_by_lr_sure_saccade[lr][w,unit],
                      color=lr_colors[in_opp],
                      linestyle='--', dashes=dashes,
                      **lineprops)
            all_y.append(Ts_states_by_lr_sure_saccade[lr][w,unit])
        for lr in Ts_states_by_lr_saccade:
            in_opp = lr*preferred_targets[unit]
            plot.plot(time_aligned[w], Ts_states_by_lr_saccade[lr][w,unit],
                      color=lr_colors[in_opp], **lineprops)
            all_y.append(Ts_states_by_lr_saccade[lr][w,unit])

        plot.xticks([-300, 0])

        plot.xlim(-500, 200)
        #plot.ylim(0, 1)
        plot.lim('y', all_y, lower=0)

        plot.vline(0, **vline_props)

    #=====================================================================================
    # Plot
    #=====================================================================================

    if unit is not None:
        noTs_dots(plots['noTs-dots'], unit)
        noTs_saccade(plots['noTs-saccade'], unit)
        Ts_dots(plots['Ts-dots'], unit)
        Ts_sure(plots['Ts-sure'], unit)
        Ts_saccade(plots['Ts-saccade'], unit)
    else:
        figspath, name = plots
        for unit in xrange(N):
            w = 174/25.4
            r = 0.35
            h = r*w
            fig = Figure(w=w, h=h)

            x0 = 0.07
            y0 = 0.15

            w = 0.13
            h = 0.75

            dx = 0.05
            DX = 0.08

            plots = {
                'noTs-dots':    fig.add([x0, y0, w, h]),
                'noTs-saccade': fig.add([x0+w+dx, y0, w, h]),
                'Ts-dots':      fig.add([x0+w+dx+w+DX, y0, w, h]),
                'Ts-sure':      fig.add([x0+w+dx+w+DX+w+dx, y0, w, h]),
                'Ts-saccade':   fig.add([x0+w+dx+w+DX+w+dx+w+dx, y0, w, h])
                }

            #-----------------------------------------------------------------------------
            # Aligned to dots onset (no Ts)
            #-----------------------------------------------------------------------------

            noTs_dots(plots['noTs-dots'], unit)

            #-----------------------------------------------------------------------------
            # Aligned to saccade (no Ts)
            #-----------------------------------------------------------------------------

            noTs_saccade(plots['noTs-saccade'], unit)

            #-----------------------------------------------------------------------------
            # Aligned to dots onset (Ts)
            #-----------------------------------------------------------------------------

            Ts_dots(plots['Ts-dots'], unit)

            #-----------------------------------------------------------------------------
            # Aligned to sure target (Ts)
            #-----------------------------------------------------------------------------

            Ts_sure(plots['Ts-sure'], unit)

            #-----------------------------------------------------------------------------
            # Aligned to saccade (Ts)
            #-----------------------------------------------------------------------------

            Ts_saccade(plots['Ts-saccade'], unit)

            #-----------------------------------------------------------------------------

            fig.save(path=figspath, name=name+'_{:03d}'.format(unit))
            fig.close()

#/////////////////////////////////////////////////////////////////////////////////////////

def performance(savefile, plot, **kwargs):
    perf_history = utils.load(savefile)['perf_history']

    all_trials    = []
    all_corrects  = []
    best_trials   = []
    best_corrects = []
    for niters, ntrials, perf, is_best in perf_history:
        if is_best:
            if perf.n_decision > 0:
                p_correct = perf.n_correct/perf.n_decision
                best_trials.append(ntrials)
                best_corrects.append(p_correct)
        if perf.n_decision > 0:
            p_correct = perf.n_correct/perf.n_decision
            all_trials.append(ntrials)
            all_corrects.append(p_correct)

    all_trials    = np.asarray(all_trials)
    all_corrects  = np.asarray(all_corrects)
    best_trials   = np.asarray(best_trials)
    best_corrects = np.asarray(best_corrects)

    M = 100

    plot.plot(all_trials/M, 100*all_corrects, color='0.8', lw=1.5)
    plot.plot(all_trials/M, 100*all_corrects, 'o', mfc='0.8', mew=0)

    plot.plot(best_trials/M, 100*best_corrects, color='k', lw=1.5)
    plot.plot(best_trials/M, 100*best_corrects, 'o', mfc='k', mew=0)

    plot.xlim(0, max(all_trials/M))
    plot.ylim(0, 100)

    plot.xlabel(r'Number of trials ($\times$ 10$^2$)')
    plot.ylabel('Percent correct')

def do(action, args, config):
    """
    Manage tasks.

    """
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    if action == 'performance':
        fig  = Figure(axislabelsize=10, ticklabelsize=9)
        plot = fig.add()

        performance(config['savefile'], plot)

        fig.save(path=config['figspath'], name='performance')
        fig.close()

    elif action in ['trials-b', 'trials-e']:
        try:
            trials_per_condition = int(args[0])
        except:
            trials_per_condition = 1000

        model = config['model']
        m     = config['model'].m
        pg    = model.get_pg(config['savefile'], config['seed'])
        rng   = np.random.RandomState(1)

        # Conditions
        if 0 in m.cohs:
            cohs = m.cohs[1:]
        else:
            cohs = m.cohs
        left_rights = m.left_rights
        if 'wager' in args:
            wagers = m.wagers
        else:
            wagers = None

        # Number of conditions, trials
        n_conditions = 1 + len(cohs)*len(left_rights)
        n_trials     = n_conditions * trials_per_condition

        print("Generating {} trial conditions ...".format(n_trials))
        trials = []
        for n in xrange(n_trials):
            c = n % n_conditions
            if c == 0:
                # Zero-coherence condition
                coh        = 0
                left_right = rng.choice(left_rights)
            else:
                # All other conditions
                k1, k2     = tasktools.unravel_index(c-1, (len(cohs), len(left_rights)))
                coh        = cohs[k1]
                left_right = left_rights[k2]
            context = {'cohs': [coh], 'left_rights': [left_right]}
            trial   = m.condition(rng, pg.config['dt'], context)

            trials.append(trial)

        print("Running trials ...")
        if action == 'trials-e':
            print("Save states.")
            name = 'trials_electrophysiology'
            #(U, Q, Z, A, R, M, init, states_0, perf,
            # states) = pg.run_trials(trials, return_states=True, progress_bar=True)
            (U, Q, Z, A, R, M, init, states_0,
             perf) = pg.run(trials, progress_bar=True)
            save = [trials, U, Q, Z, A, R, M, perf, states]
        else:
            print("Behavior only.")
            name = 'trials_psychophysics'
            #(U, Q, Z, A, R, M, init, states_0, perf), states = pg.run_trials(trials, progress_bar=True), None
            #save = [trials, A, R, M, perf]
            (U, Q, Z, A, R, M, init, states_0, perf,
             states, baseline_states) = pg.run(trials, return_states=True, return_baseline_states=True, progress_bar=True)
            save = [trials, U, Q, Z, A, R, M, perf, states, baseline_states]
        perf.display()

        # Save
        print("Saving ...")
        trialsfile = os.path.join(config['scratchpath'], name + '.pkl')
        utils.save(trialsfile, save)

        # File size
        size_in_bytes = os.path.getsize(trialsfile)
        print("File size: {:.1f} MB".format(size_in_bytes/2**20))

    elif action == 'psychometric':
        trialsfile = os.path.join(config['scratchpath'], 'trials_psychophysics.pkl')

        fig  = Figure(axislabelsize=10, ticklabelsize=9)
        plot = fig.add()

        psychometric(trialsfile, config['model'].m, plot, 'psychometric')

        fig.save(path=config['figspath'], name='psychometric')
        fig.close()

    elif action == 'chronometric':
        trialsfile = os.path.join(config['scratchpath'], 'trials_psychophysics.pkl')

        fig  = Figure(axislabelsize=10, ticklabelsize=9)
        plot = fig.add()

        chronometric(trialsfile, config['figspath'], 'chronometric')

        plot.xlabel('\% Coherence')
        plot.ylabel('Reaction time (ms)')

        fig.save(path=config['figspath'], name='chronometric')
        fig.close()

    elif action == 'sort-rt':
        trialsfile = os.path.join(config['scratchpath'], 'trials_electrophysiology.pkl')
        sort_rt(trialsfile, config['figspath'], 'sorted_rt')

    elif action == 'sort-postdecision':
        trialsfile = os.path.join(config['scratchpath'], 'trials_electrophysiology.pkl')
        sort_postdecision(trialsfile, config['model'].m,
                          (config['figspath'], 'sorted_postdecision'))

    elif action == 'sure-stimulus-duration':
        trialsfile = os.path.join(config['scratchpath'], 'trials_psychophysics.pkl')

        fig  = Figure(axislabelsize=10, ticklabelsize=9)
        plot = fig.add()

        sure_stimulus_duration(trialsfile, plot, nbins=10)

        fig.save(path=config['figspath'], name='sure_stimulus_duration')
        fig.close()

    elif action == 'stimulus-duration':
        wager = ('wager' in args)

        trialsfile = os.path.join(config['scratchpath'], 'trials_psychophysics.pkl')

        fig  = Figure(axislabelsize=10, ticklabelsize=9)
        plot = fig.add()

        stimulus_duration(trialsfile, plot, wager=wager, nbins=10)

        plot.xlabel('Stimulus duration (ms)')
        plot.ylabel('Percent correct')

        fig.save(path=config['figspath'], name='stimulus_duration')
        fig.close()
