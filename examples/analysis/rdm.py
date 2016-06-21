from __future__ import division

import os

import numpy as np

from pyrl          import datatools, fittools, runtools, tasktools, utils
from pyrl.figtools import Figure

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

def plot_trial(n, trialsfile, plots, unit=None, network='policy', **kwargs):
    # Load trials
    trials, U, Z, A, rho, M, perf, r_policy, r_value = utils.load(trialsfile)

    trial = trials[n]
    U     = U[:,n]
    Z     = Z[:,n]
    A     = A[:,n]
    rho   = rho[:,n]
    M     = M[:,n]
    tmax  = int(np.sum(M))

    # Which network?
    if network == 'policy':
        r = r_policy[:,n]
    else:
        r = r_value[:,n]

    # Data shape
    Ntime = r.shape[0]
    N     = r.shape[-1]

    w = 0.65
    h = 0.18
    x = 0.17
    dy = h + 0.05
    y0 = 0.08
    y1 = y0 + dy
    y2 = y1 + dy
    y3 = y2 + dy

    figspath, name = plots

    fig = Figure(h=6)
    fig.add('observables', [x, y3, w, h])
    fig.add('policy',      [x, y2, w, h])
    fig.add('actions',     [x, y1, w, h])
    fig.add('rewards',     [x, y0, w, h])

    time        = trial['time']
    act_time    = time[:tmax]
    obs_time    = time[1:1+tmax]
    reward_time = time[1:1+tmax]
    xlim        = (0, time[1+tmax])

    #-------------------------------------------------------------------------------------
    # Observables
    #-------------------------------------------------------------------------------------

    plot = fig['observables']
    plot.plot(obs_time, U[:tmax,0], 'o', ms=5, mew=0, mfc=Figure.colors('blue'))
    plot.plot(obs_time, U[:tmax,0], lw=1.25, color=Figure.colors('blue'),   label='Fixation')
    plot.plot(obs_time, U[:tmax,1], 'o', ms=5, mew=0, mfc=Figure.colors('orange'))
    plot.plot(obs_time, U[:tmax,1], lw=1.25, color=Figure.colors('orange'), label='Left')
    plot.plot(obs_time, U[:tmax,2], 'o', ms=5, mew=0, mfc=Figure.colors('purple'))
    plot.plot(obs_time, U[:tmax,2], lw=1.25, color=Figure.colors('purple'), label='Right')
    try:
        plot.plot(obs_time, U[:tmax,3], 'o', ms=5, mew=0, mfc=Figure.colors('green'))
        plot.plot(obs_time, U[:tmax,3], lw=1.25, color=Figure.colors('green'), label='Sure')
    except IndexError:
        pass

    plot.xlim(*xlim)
    plot.ylim(0, 1)
    plot.ylabel('Observables')

    coh = trial['left_right']*trial['coh']
    if coh < 0:
        color = Figure.colors('orange')
    elif coh > 0:
        color = Figure.colors('purple')
    else:
        color = Figure.colors('k')
    plot.text_upper_right('Coh = {:.1f}\%'.format(coh), color=color)

    props = {'prop': {'size': 7}, 'handlelength': 1.2,
             'handletextpad': 1.2, 'labelspacing': 0.8}
    plot.legend(bbox_to_anchor=(1.2, 0.8), **props)

    #plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------
    # Policy
    #-------------------------------------------------------------------------------------

    plot = fig['policy']
    plot.plot(act_time, Z[:tmax,0], 'o', ms=5, mew=0, mfc=Figure.colors('blue'))
    plot.plot(act_time, Z[:tmax,0], lw=1.25, color=Figure.colors('blue'),
              label='Fixate')
    plot.plot(act_time, Z[:tmax,1], 'o', ms=5, mew=0, mfc=Figure.colors('orange'))
    plot.plot(act_time, Z[:tmax,1], lw=1.25, color=Figure.colors('orange'),
              label='Saccade LEFT')
    plot.plot(act_time, Z[:tmax,2], 'o', ms=5, mew=0, mfc=Figure.colors('purple'))
    plot.plot(act_time, Z[:tmax,2], lw=1.25, color=Figure.colors('purple'),
              label='Saccade RIGHT')
    try:
        plot.plot(act_time, Z[:tmax,3], 'o', ms=5, mew=0, mfc=Figure.colors('green'))
        plot.plot(act_time, Z[:tmax,3], lw=1.25, color=Figure.colors('green'),
                  label='Saccade SURE')
    except IndexError:
        pass

    plot.xlim(*xlim)
    plot.ylim(0, 1)
    plot.ylabel('Action probabilities')

    props = {'prop': {'size': 7}, 'handlelength': 1.2,
             'handletextpad': 1.2, 'labelspacing': 0.8}
    plot.legend(bbox_to_anchor=(1.27, 0.8), **props)

    #plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------
    # Actions
    #-------------------------------------------------------------------------------------

    plot = fig['actions']
    actions = [np.argmax(a) for a in A[:tmax]]
    plot.plot(act_time, actions, 'o', ms=5, mew=0, mfc=Figure.colors('red'))
    plot.plot(act_time, actions, lw=1.25, color=Figure.colors('red'))
    plot.xlim(*xlim)
    yticklabels = ['Fixate', 'Saccade LEFT', 'Saccade RIGHT']
    if A.shape[1] == 4:
        yticklabels += ['Saccade sure']
    plot.yticklabels(yticklabels)
    plot.ylim(0, len(yticklabels)-1)
    plot.yticks(range(len(yticklabels)))

    plot.ylabel('Action')

    #plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------
    # Rewards
    #-------------------------------------------------------------------------------------

    plot = fig['rewards']
    plot.plot(reward_time, rho[:tmax], 'o', ms=5, mew=0, mfc=Figure.colors('red'))
    plot.plot(reward_time, rho[:tmax], lw=1.25, color=Figure.colors('red'))

    # Prediction
    #if b is not None:
    #    plot.plot(reward_time, b[:tmax], 'o', ms=5, mew=0, mfc=Figure.colors('orange'))
    #    plot.plot(reward_time, b[:tmax], lw=1.25, color=Figure.colors('orange'))

    plot.xlim(*xlim)
    #plot.ylim(m.R_ABORTED, m.R_CORRECT)
    plot.xlabel('Time (ms)')
    plot.ylabel('Reward')

    #plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------

    fig.save(path=figspath, name=name)
    fig.close()

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

            if perf.choices[n] == 'R':
                right = 1
            elif perf.choices[n] == 'L':
                right = 0
            else:
                raise ValueError("invalid choice")

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

    # Time
    time = trials[0]['time']

    zero_coherence_rt = []
    correct_rt_by_coh = {}
    error_rt_by_coh   = {}
    no_decision       = 0
    for n, trial in enumerate(trials):
        coh = trial['coh']
        if coh == 0:
            if perf.decisions[n]:
                stimulus_start = trial['durations']['stimulus'][0]
                rt = time[perf.t_choices[n]] - stimulus_start
                zero_coherence_rt.append(rt)
            continue

        if perf.decisions[n]:
            stimulus_start = trial['durations']['stimulus'][0]
            rt = time[perf.t_choices[n]] - stimulus_start
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

#/////////////////////////////////////////////////////////////////////////////////////////

def sort(trialsfile, plots, unit=None, network='p', **kwargs):
    """
    Sort trials.

    """
    # Load trials
    trials, U, Z, A, P, M, perf, r_p, r_v = utils.load(trialsfile)

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

        # Signed coherence
        coh = trial['coh']

        # Choice
        if perf.choices[n] == 'R':
            choice = 1
        else:
            choice = 0

        # Condition
        cond = (coh, choice)

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

    #=====================================================================================
    # Aligned to choice
    #=====================================================================================

    r_by_cond_choice   = {}
    n_r_by_cond_choice = {}
    for n, trial in enumerate(trials):
        if not perf.decisions[n]:
            continue

        # Signed coherence
        coh = trial['coh']

        # Choice
        if perf.choices[n] == 'R':
            choice = 1
        else:
            choice = 0

        # Condition
        cond = (coh, choice)

        # Storage
        r_by_cond_choice.setdefault(cond, np.zeros((Ntime_a, N)))
        n_r_by_cond_choice.setdefault(cond, np.zeros((Ntime_a, N)))

        # Firing rates
        Mn = np.tile(M[:,n], (N,1)).T
        Rn = r[:,n]*Mn

        # Align point
        t0 = perf.t_choices[n]
        #np.where(time <= perf.t_choices[n])[0][-1]
        #assert t0 == np.sum(M[:,n])-1

        # Before
        n_b = Rn[:t0].shape[0]
        r_by_cond_choice[cond][Ntime-1-n_b:Ntime-1]   += Rn[:t0]
        n_r_by_cond_choice[cond][Ntime-1-n_b:Ntime-1] += Mn[:t0]

        # After
        n_a = Rn[t0:].shape[0]
        r_by_cond_choice[cond][Ntime-1:Ntime-1+n_a]   += Rn[t0:]
        n_r_by_cond_choice[cond][Ntime-1:Ntime-1+n_a] += Mn[t0:]

    for cond in r_by_cond_choice:
        r_by_cond_choice[cond] = utils.div(r_by_cond_choice[cond],
                                           n_r_by_cond_choice[cond])

    #=====================================================================================
    # Plot
    #=====================================================================================

    lw     = kwargs.get('lw', 1.5)
    dashes = kwargs.get('dashes', [4, 1.5])

    time = trials[0]['time']

    lr_colors = {1: '0', -1: '0.5'}

    def plot_sorted(plot, unit, w, r_sorted, clrs=colors):
        t = time_a[w]
        yall = [[1]]
        for cond in sorted(r_sorted.keys()):
            coh, choice = cond
            if choice > 0:
                props = dict(color=clrs[coh], label='{}\%'.format(coh))
            else:
                props = dict(color=clrs[coh])

            if choice > 0:
                ls = '-'
                plot.plot(t, r_sorted[cond][w,unit], ls=ls, lw=lw, **props)
            else:
                ls = '--'
                plot.plot(t, r_sorted[cond][w,unit], ls=ls, lw=lw, dashes=dashes, **props)
            yall.append(r_sorted[cond][w,unit])

        return t, yall

    def on_stimulus(plot, unit, tmin=-200, tmax=1000, clrs=colors):
        w, = np.where((tmin <= time_a) & (time_a <= tmax))
        t, yall = plot_sorted(plot, unit, w, r_by_cond_stimulus, clrs)

        plot.xlim(t[0], t[-1])

        plot.lim('y', yall, lower=0)

        return yall

    def on_choice(plot, unit, tmin=-600, tmax=100):
        w, = np.where((time_a >= tmin) & (time_a <= tmax))
        t, yall = plot_sorted(plot, unit, w, r_by_cond_choice)

        plot.xlim(t[0], t[-1])

        return yall

    if unit is not None:
        if kwargs.get('colors') is None:
            clrs = colors
        else:
            clrs = colors_kiani2009

        tmin = kwargs.get('on-stimulus-tmin', -200)
        tmax = kwargs.get('on-stimulus-tmax', 1000)
        if 'on-stimulus' in plots:
            on_stimulus(plots['on-stimulus'], unit, tmin, tmax, clrs)

        if 'on-choice' in plots:
            on_choice(plots['on-choice'], unit)
    else:
        name = plots
        for unit in xrange(N):
            w = 174/25.4
            r = 0.5
            h = r*w
            fig = Figure(w=w, h=h)

            x0 = 0.12
            y0 = 0.15
            w  = 0.37
            h  = 0.8
            dx = 1.3*w

            fig.add('on-stimulus', [x0,    y0, w, h])
            fig.add('on-choice',   [x0+dx, y0, w, h])

            #-----------------------------------------------------------------------------

            yall = []
            yall += on_stimulus(fig['on-stimulus'], unit)
            yall += on_choice(fig['on-choice'], unit)

            plot = fig['on-stimulus']
            plot.lim('y', yall, lower=0)
            plot.vline(0)
            plot.xlabel('Time (ms)')
            plot.ylabel('Firing rate (a.u.)')

            plot = fig['on-choice']
            plot.lim('y', yall, lower=0)
            plot.vline(0)

            #-----------------------------------------------------------------------------

            fig.save(name+'_{}{:03d}'.format(network, unit))
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

def correct_stimulus_duration(trialsfile, plot, wager=False, saved=None, **kwargs):
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

def get_choice_selectivity(trials, perf, r):
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
        if not perf.decisions[n]:
            continue

        stimulus = trial['epochs']['stimulus']
        r_n      = r[stimulus,n]

        left_right = trial['left_right']

        if left_right < 0:
            L  += np.sum(r_n,    axis=0)
            L2 += np.sum(r_n**2, axis=0)
            nL += len(stimulus)
        else:
            R  += np.sum(r_n,    axis=0)
            R2 += np.sum(r_n**2, axis=0)
            nR += len(stimulus)

    mean_L = L/nL
    var_L  = L2/nL - mean_L**2

    mean_R = R/nR
    var_R  = R2/nR - mean_R**2

    return -utils.div(mean_L - mean_R, np.sqrt((var_L + var_R)/2))

def get_preferred_targets(trials, perf, r):
    """
    Determine preferred targets.

    """
    dprime = get_choice_selectivity(trials, perf, r)
    for i in xrange(len(dprime)):
        print(i, dprime[i])

    return 2*(dprime > 0) - 1

def sort_postdecision(trialsfile, m, plots, unit=None, **kwargs):
    """
    Sort postdecision trials by condition.

    """
    # Load trials
    trials, U, Z, A, R, M, perf, states, baseline_states = utils.load(trialsfile)

    # Data shape
    Ntime = states.shape[0]
    N     = states.shape[-1]

    # Same for every trial
    time = trials[0]['time']

    # Aligned time
    time_aligned = np.concatenate((-time[1:][::-1], time))

    #=====================================================================================
    # Preferred targets
    #=====================================================================================

    preferred_targets = get_preferred_targets(trials, perf, states)

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

            fig.add('noTs-dots',    [x0, y0, w, h])
            fig.add('noTs-saccade', [x0+w+dx, y0, w, h])
            fig.add('Ts-dots',      [x0+w+dx+w+DX, y0, w, h])
            fig.add('Ts-sure',      [x0+w+dx+w+DX+w+dx, y0, w, h])
            fig.add('Ts-saccade',   [x0+w+dx+w+DX+w+dx+w+dx, y0, w, h])

            #-----------------------------------------------------------------------------
            # Aligned to dots onset (no Ts)
            #-----------------------------------------------------------------------------

            noTs_dots(fig['noTs-dots'], unit)

            #-----------------------------------------------------------------------------
            # Aligned to saccade (no Ts)
            #-----------------------------------------------------------------------------

            noTs_saccade(fig['noTs-saccade'], unit)

            #-----------------------------------------------------------------------------
            # Aligned to dots onset (Ts)
            #-----------------------------------------------------------------------------

            Ts_dots(fig['Ts-dots'], unit)

            #-----------------------------------------------------------------------------
            # Aligned to sure target (Ts)
            #-----------------------------------------------------------------------------

            Ts_sure(fig['Ts-sure'], unit)

            #-----------------------------------------------------------------------------
            # Aligned to saccade (Ts)
            #-----------------------------------------------------------------------------

            Ts_saccade(fig['Ts-saccade'], unit)

            #-----------------------------------------------------------------------------

            fig.save(path=figspath, name=name+'_{:03d}'.format(unit))
            fig.close()

#/////////////////////////////////////////////////////////////////////////////////////////

def training_history(savefile, plot, **kwargs):
    training_history = utils.load(savefile)['training_history']

    all_trials    = []
    all_corrects  = []
    best_trials   = []
    best_corrects = []
    for record in training_history:
        niters      = record['iter']
        mean_reward = record['mean_reward']
        ntrials     = record['n_trials']
        perf        = record['perf']
        is_best     = record['new_best']
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
    plot.ylim(40, 100)

    plot.xlabel(r'Number of trials ($\times$ 10$^2$)')
    plot.ylabel('Percent correct')

#/////////////////////////////////////////////////////////////////////////////////////////

def do(action, args, config):
    """
    Manage tasks.

    """
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    #=====================================================================================

    if action == 'training_history':
        fig  = Figure()
        plot = fig.add()

        training_history(config['savefile'], plot)

        fig.save(os.path.join(config['figspath'], action))
        fig.close()

    #=====================================================================================

    elif 'trials' in action:
        try:
            trials_per_condition = int(args[0])
        except:
            trials_per_condition = 100

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'], config['dt'])

        spec         = model.spec
        left_rights  = spec.left_rights
        cohs         = spec.cohs
        n_conditions = spec.n_conditions
        n_trials     = trials_per_condition * n_conditions

        print("{} trials".format(n_trials))
        task   = model.Task()
        trials = []
        for n in xrange(n_trials):
            k = tasktools.unravel_index(n, (len(left_rights), len(cohs)))
            context = {
                'left_right': left_rights[k.pop(0)],
                'coh':        cohs[k.pop(0)]
                }
            trials.append(task.get_condition(pg.rng, pg.dt, context))
        runtools.run(action, trials, pg, config['trialspath'])

    elif action == 'psychometric':
        trialsfile = runtools.behaviorfile(config['trialspath'])

        fig  = Figure()
        plot = fig.add()

        psychometric(trialsfile, config['model'].spec, plot, 'psychometric')

        fig.save(path=config['figspath'], name='psychometric')
        fig.close()

    elif action == 'chronometric':
        trialsfile = runtools.behaviorfile(config['trialspath'])

        fig  = Figure()
        plot = fig.add()

        chronometric(trialsfile, plot)

        plot.xlabel('\% Coherence')
        plot.ylabel('Reaction time (ms)')

        fig.save(path=config['figspath'], name='chronometric')
        fig.close()

    elif action == 'plot-trial':
        n = int(args[0])

        if 'value' in args:
            network = 'value'
        else:
            network = 'policy'

        trialsfile = os.path.join(config['trialspath'], 'trials_activity.pkl')
        plot_trial(n, trialsfile, (config['figspath'], 'trial{}_{}'.format(n, network)),
                   network=network)

    elif action == 'sort':
        if 'value' in args:
            network = 'v'
        else:
            network = 'p'

        sort(runtools.activityfile(config['trialspath']),
             os.path.join(config['figspath'], 'sorted'),
             network=network)

    elif action == 'sort-postdecision':
        trialsfile = os.path.join(config['trialspath'], 'trials_activity.pkl')
        sort_postdecision(trialsfile, config['model'].spec,
                          (config['figspath'], 'sorted_postdecision'))

    elif action == 'sure-stimulus-duration':
        trialsfile = os.path.join(config['trialspath'], 'trials_behavior.pkl')

        fig  = Figure()
        plot = fig.add()

        sure_stimulus_duration(trialsfile, plot, nbins=10)

        fig.save(path=config['figspath'], name='sure_stimulus_duration')
        fig.close()

    elif action == 'correct_stimulus_duration':
        wager = ('wager' in args)

        trialsfile = runtools.behaviorfile(config['trialspath'])

        fig  = Figure()
        plot = fig.add()

        correct_stimulus_duration(trialsfile, plot, wager=wager, nbins=10)

        plot.xlabel('Stimulus duration (ms)')
        plot.ylabel('Percent correct')

        fig.save(os.path.join(config['figspath'], action))
        fig.close()
