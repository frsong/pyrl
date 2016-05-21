from __future__ import absolute_import, division

import os

import numpy as np

from pyrl                import tasktools
from pyrl                import utils
from pyrl.policygradient import PolicyGradient

from pycog           import fittools
from pycog.datatools import partition
from pycog.figtools  import Figure

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

def plot_trial(pg, m, init, init_b, rng, figspath, name):
    context = {}
    if 0 not in m.cohs:
        context['cohs'] = [0] + m.cohs
    trial = m.generate_trial_condition(rng, context)

    U, Q, Z, A, R, M, init, states_0, perf = pg.run_trials([trial], init=init)
    if pg.baseline_net is not None:
        (init_b, baseline_states_0, b,
         rpe) = pg.baseline_run_trials(U, Q, A, R, M, init=init_b)
    else:
        b = None

    U = U[:,0,:]
    Z = Z[:,0,:]
    A = A[:,0,:]
    R = R[:,0]
    M = M[:,0]
    t = int(np.sum(M))

    w = 0.65
    h = 0.18
    x = 0.17
    dy = h + 0.05
    y0 = 0.08
    y1 = y0 + dy
    y2 = y1 + dy
    y3 = y2 + dy

    fig   = Figure(h=6)
    plots = {'observables': fig.add([x, y3, w, h]),
             'policy':      fig.add([x, y2, w, h]),
             'actions':     fig.add([x, y1, w, h]),
             'rewards':     fig.add([x, y0, w, h])}

    time        = trial['time']
    dt          = time[1] - time[0]
    act_time    = time[:t]
    obs_time    = time[:t] + dt
    reward_time = act_time + dt
    xlim        = (0, max(time))

    #-------------------------------------------------------------------------------------
    # Observables
    #-------------------------------------------------------------------------------------

    plot = plots['observables']
    plot.plot(obs_time, U[:t,0], 'o', ms=5, mew=0, mfc=Figure.colors('blue'))
    plot.plot(obs_time, U[:t,0], lw=1.25, color=Figure.colors('blue'),   label='Fixation')
    plot.plot(obs_time, U[:t,1], 'o', ms=5, mew=0, mfc=Figure.colors('orange'))
    plot.plot(obs_time, U[:t,1], lw=1.25, color=Figure.colors('orange'), label='Left')
    plot.plot(obs_time, U[:t,2], 'o', ms=5, mew=0, mfc=Figure.colors('purple'))
    plot.plot(obs_time, U[:t,2], lw=1.25, color=Figure.colors('purple'), label='Right')
    try:
        plot.plot(obs_time, U[:t,3], 'o', ms=5, mew=0, mfc=Figure.colors('green'))
        plot.plot(obs_time, U[:t,3], lw=1.25, color=Figure.colors('green'), label='Sure')
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

    plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------
    # Policy
    #-------------------------------------------------------------------------------------

    plot = plots['policy']
    plot.plot(act_time, Z[:t,0], 'o', ms=5, mew=0, mfc=Figure.colors('blue'))
    plot.plot(act_time, Z[:t,0], lw=1.25, color=Figure.colors('blue'),
              label='Fixate')
    plot.plot(act_time, Z[:t,1], 'o', ms=5, mew=0, mfc=Figure.colors('orange'))
    plot.plot(act_time, Z[:t,1], lw=1.25, color=Figure.colors('orange'),
              label='Saccade LEFT')
    plot.plot(act_time, Z[:t,2], 'o', ms=5, mew=0, mfc=Figure.colors('purple'))
    plot.plot(act_time, Z[:t,2], lw=1.25, color=Figure.colors('purple'),
              label='Saccade RIGHT')
    try:
        plot.plot(act_time, Z[:t,3], 'o', ms=5, mew=0, mfc=Figure.colors('green'))
        plot.plot(act_time, Z[:t,3], lw=1.25, color=Figure.colors('green'),
                  label='Saccade SURE')
    except IndexError:
        pass

    plot.xlim(*xlim)
    plot.ylim(0, 1)
    plot.ylabel('Action probabilities')

    props = {'prop': {'size': 7}, 'handlelength': 1.2,
             'handletextpad': 1.2, 'labelspacing': 0.8}
    plot.legend(bbox_to_anchor=(1.27, 0.8), **props)

    plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------
    # Actions
    #-------------------------------------------------------------------------------------

    plot = plots['actions']
    actions = [np.argmax(a) for a in A[:t]]
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

    plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------
    # Rewards
    #-------------------------------------------------------------------------------------

    plot = plots['rewards']
    plot.plot(reward_time, R[:t], 'o', ms=5, mew=0, mfc=Figure.colors('red'))
    plot.plot(reward_time, R[:t], lw=1.25, color=Figure.colors('red'))

    # Prediction
    if b is not None:
        plot.plot(reward_time, b[:t], 'o', ms=5, mew=0, mfc=Figure.colors('orange'))
        plot.plot(reward_time, b[:t], lw=1.25, color=Figure.colors('orange'))

    plot.xlim(*xlim)
    plot.ylim(m.R_ABORTED, m.R_CORRECT)
    plot.xlabel('Time (ms)')
    plot.ylabel('Reward')

    plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------

    fig.save(path=figspath, name=name)
    fig.close()

    #-------------------------------------------------------------------------------------

    return init, init_b

def plot_psychometric(pg, rng, m, n_trials, figspath, name, **kwargs):
    #-------------------------------------------------------------------------------------
    # Run trials
    #-------------------------------------------------------------------------------------

    if m.cohs[0] == 0:
        cohs_ = m.cohs[1:]
    else:
        cohs_ = m.cohs
    signed_cohs = sorted([-coh for coh in cohs_]) + [0] + sorted([+coh for coh in cohs_])

    conditions = [(np.sign(signed_coh), np.abs(signed_coh)) for signed_coh in signed_cohs]

    n_trials_by_c   = {c: 0 for c in conditions}
    n_decision_by_c = {c: 0 for c in conditions}
    n_right_by_c    = {c: 0 for c in conditions}

    init = None

    backspace = 0
    for n in xrange(n_trials):
        s = '{}/{}'.format(n+1, n_trials)
        utils.println(backspace*'\b' + s)
        backspace = len(s)

        signed_coh = signed_cohs[n%len(signed_cohs)]
        left_right = np.sign(signed_coh)
        coh        = np.abs(signed_coh)

        context = {'left_rights': [left_right], 'cohs': [coh]}
        trial   = m.generate_trial_condition(rng, context)
        U, Q, Z, A, R, M, init, states_0, perf = pg.run_trials([trial], init=init)
        A = A[:,0,:]
        M = M[:,0]

        if 'actions_map' in vars(m):
            actions_map = m.actions_map
        else:
            actions_map = m.actions

        c = (left_right, coh)
        if perf.n_decision == 1:
            n_decision_by_c[c] += 1
            t = int(np.sum(M))-1
            if A[t,actions_map['SACCADE_RIGHT']] == 1:
                n_right_by_c[c] += 1
        n_trials_by_c[c] += 1
    print("")

    def safe_divide(x, y):
        if y == 0:
            return 0
        return x/y

    p_decision = [safe_divide(n_decision_by_c[c], n_trials_by_c[c]) for c in conditions]
    p_right    = [safe_divide(n_right_by_c[c], n_decision_by_c[c])  for c in conditions]

    #-------------------------------------------------------------------------------------
    # Psychometric curve
    #-------------------------------------------------------------------------------------

    fig  = Figure()
    plot = fig.add()

    #-------------------------------------------------------------------------------------

    plot.plot(signed_cohs, p_decision, lw=1.5, color=Figure.colors('orange'),
              label='$P$(decision)')
    plot.plot(signed_cohs, p_decision, 'o', ms=6, mew=0, mfc=Figure.colors('orange'))

    # Fit psychometric curve
    props = dict(lw=1.5, color=Figure.colors('blue'), label='$P$(right$|$decision)')
    try:
        popt, func = fittools.fit_psychometric(signed_cohs, p_right)

        fit_cohs = np.linspace(min(signed_cohs), max(signed_cohs), 201)
        fit_pr   = func(fit_cohs, **popt)
        plot.plot(fit_cohs, fit_pr, **props)
    except RuntimeError:
        print("Unable to fit, drawing a line through the points.")
        plot.plot(signed_cohs, p_right, **props)
    plot.plot(signed_cohs, p_right, 'o', ms=6, mew=0, mfc=Figure.colors('blue'))

    plot.xlim(signed_cohs[0], signed_cohs[-1])
    plot.ylim(0, 1)

    plot.hline(0.5, linestyle='--')

    plot.xlabel('Percent coherence toward right')
    plot.ylabel('Prop. decision/Prop. right')

    props = {'prop': {'size': 8.5}, 'handlelength': 1.4,
             'handletextpad': 1.2, 'labelspacing': 0.9}
    plot.legend(bbox_to_anchor=(0.97, 0.17), **props)

    #-------------------------------------------------------------------------------------

    fig.save(path=figspath, name=name)
    fig.close()

def psychometric(trialsfile, m, figspath, name, **kwargs):
    # Load trials
    trials, U, Q, Z, A, R, M, perf, states = utils.load(trialsfile)

    if 'actions_map' in vars(m):
        actions_map = m.actions_map
    else:
        actions_map = m.actions

    decision_by_coh = {}
    right_by_coh    = {}
    for n, trial in enumerate(trials):
        coh = trial['left_right']*trial['coh']
        decision_by_coh.setdefault(coh, [])
        if perf.decisions[n]:
            decision_by_coh[coh].append(1)

            t = int(np.sum(M[:,n]))-1
            if A[t,n,actions_map['SACCADE_RIGHT']] == 1:
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
    # Psychometric curve
    #-------------------------------------------------------------------------------------

    fig  = Figure()
    plot = fig.add()

    #-------------------------------------------------------------------------------------

    # Prop. decision
    plot.plot(cohs, 100*p_decision, lw=1.5, color=Figure.colors('orange'),
              label='$P$(decision)')
    plot.plot(cohs, 100*p_decision, 'o', ms=6, mew=0, mfc=Figure.colors('orange'))

    # Fit psychometric curve
    props = dict(lw=1.5, color=Figure.colors('blue'), label='$P$(right$|$decision)')
    try:
        popt, func = fittools.fit_psychometric(cohs, p_right)

        fit_cohs = np.linspace(min(cohs), max(cohs), 201)
        fit_pr   = func(fit_cohs, **popt)
        plot.plot(fit_cohs, 100*fit_pr, **props)
    except RuntimeError:
        print("Unable to fit, drawing a line through the points.")
        plot.plot(cohs, 100*p_right, **props)
    plot.plot(cohs, 100*p_right, 'o', ms=6, mew=0, mfc=Figure.colors('blue'))

    plot.xlim(cohs[0], cohs[-1])
    plot.ylim(0, 100)

    plot.hline(50, linestyle='--')

    plot.xlabel('Percent coherence toward right')
    plot.ylabel('Percent decision/right')

    props = {'prop': {'size': 8.5}, 'handlelength': 1.4,
             'handletextpad': 1.2, 'labelspacing': 0.9}
    plot.legend(bbox_to_anchor=(0.97, 0.17), **props)

    #-------------------------------------------------------------------------------------

    fig.save(path=figspath, name=name)
    fig.close()

def plot_stimulus_duration(trialsfile, figspath, name, **kwargs):
    """
    Percent correct as a function of stimulus duration.

    """
    # Load trials
    trials, U, Q, Z, A, R, M, perf, states = utils.load(trialsfile)

    nbins = kwargs.get('nbins', 5)

    correct_duration_by_coh = {}
    for n, trial in enumerate(trials):
        coh = trial['coh']
        if coh == 0:
            continue

        if perf.decisions[n]:
            correct_duration_by_coh.setdefault(coh, ([], []))[0].append(perf.corrects[n])
            correct_duration_by_coh[coh][1].append(trial['stimulus'])

    correct_by_coh = {}
    for coh, (correct, duration) in correct_duration_by_coh.items():
        Xbins, Ybins, Xedges, _ = partition(np.asarray(duration), np.asarray(correct),
                                            nbins=nbins)
        correct_by_coh[coh] = ((Xedges[:-1] + Xedges[1:])/2,
                               [100*utils.divide(np.sum(Ybin > 0), len(Ybin))
                                for Ybin in Ybins])

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    fig  = Figure()
    plot = fig.add()

    #-------------------------------------------------------------------------------------

    lineprop = {'lw':  kwargs.get('lw', 1)}
    dataprop = {'ms':  kwargs.get('ms', 6),
                'mew': kwargs.get('mew', 0)}

    # Nice colors to represent coherences, from http://colorbrewer2.org/
    colors = {
        0:    '#c6dbef',
        3.2:  '#9ecae1',
        6.4:  '#6baed6',
        12.8: '#4292c6',
        25.6: '#2171b5',
        51.2: '#084594'
        }

    cohs = sorted(correct_by_coh)
    xall = []
    for coh in cohs:
        stim, correct = correct_by_coh[coh]

        plot.plot(stim, correct, color=colors[coh], label='{}\%'.format(coh),
                  **lineprop)
        plot.plot(stim, correct, 'o', mfc=colors[coh], **dataprop)
        xall.append(stim)

    plot.lim('x', xall)
    plot.ylim(50, 100)

    plot.xlabel('Stimulus duration (ms)')
    plot.ylabel('Percent correct')

    #-------------------------------------------------------------------------------------

    fig.save(path=figspath, name=name)
    fig.close()

def plot_stimulus_duration_wager(trialsfile, plot, **kwargs):
    """
    Percent correct as a function of stimulus duration when there are wager trials.

    """
    # Load trials
    trials, U, Q, Z, A, R, M, perf, states = utils.load(trialsfile)

    correct_duration_by_coh        = {}
    correct_duration_by_coh_waived = {}
    for n, trial in enumerate(trials):
        coh = trial['coh']
        if coh == 0:
            continue

        if perf.decisions[n]:
            if trial['wager']:
                correct_duration_by_coh_waived.setdefault(coh, ([], []))[0].append(perf.corrects[n])
                correct_duration_by_coh_waived[coh][1].append(trial['stimulus'])
            else:
                correct_duration_by_coh.setdefault(coh, ([], []))[0].append(perf.corrects[n])
                correct_duration_by_coh[coh][1].append(trial['stimulus'])

    nbins = kwargs.get('nbins', 5)

    # No-wager trials
    correct_by_coh = {}
    for coh, (correct, duration) in correct_duration_by_coh.items():
        Xbins, Ybins, Xedges, _ = partition(np.asarray(duration), np.asarray(correct),
                                            nbins=nbins)
        correct_by_coh[coh] = ((Xedges[:-1] + Xedges[1:])/2,
                               [100*utils.divide(np.sum(Ybin > 0), len(Ybin))
                                for Ybin in Ybins])

    # Sure bet presented but waived
    correct_by_coh_waived = {}
    for coh, (correct, duration) in correct_duration_by_coh_waived.items():
        Xbins, Ybins, Xedges, _ = partition(np.asarray(duration), np.asarray(correct),
                                            nbins=nbins)
        correct_by_coh_waived[coh] = ((Xedges[:-1] + Xedges[1:])/2,
                                      [100*utils.divide(np.sum(Ybin > 0), len(Ybin))
                                       for Ybin in Ybins])

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    lineprop        = {'lw':  kwargs.get('lw', 1)}
    dataprop_waived = {'ms':  kwargs.get('ms', 6),
                       'mew': kwargs.get('mew', 0)}
    dataprop        = {'ms':  kwargs.get('ms', 6-1),
                       'mew': kwargs.get('mew', 1)}

    # Colors
    colors = colors_kiani2009

    # To determine x-limits
    xall = []

    # No-wager trials
    cohs = sorted(correct_by_coh)
    for coh in cohs:
        stim, correct = correct_by_coh[coh]

        plot.plot(stim, correct, '--', color=colors[coh], label='{}\%'.format(coh),
                  **lineprop)
        plot.plot(stim, correct, 'o', mfc='w', mec=colors[coh], **dataprop)
        xall.append(stim)

    # Sure bet presented but waived
    cohs = sorted(correct_by_coh_waived)
    for coh in cohs:
        stim, correct = correct_by_coh_waived[coh]

        plot.plot(stim, correct, color=colors[coh], label='{}\%'.format(coh),
                  **lineprop)
        plot.plot(stim, correct, 'o', mfc=colors[coh], **dataprop_waived)
        xall.append(stim)

    #plot.lim('x', xall)
    plot.xlim(100, 800)
    plot.ylim(50, 100)

    plot.xlabel('Stimulus duration (ms)')
    plot.ylabel('Percent correct')

def plot_sure_stimulus_duration(trialsfile, plot, **kwargs):
    """
    Percent correct as a function of stimulus duration.

    """
    # Load trials
    trials, U, Q, Z, A, R, M, perf, states = utils.load(trialsfile)

    sure_duration_by_coh = {}
    for n, trial in enumerate(trials):
        if not trial['wager']:
            continue

        coh = trial['coh']
        if perf.sure_decisions[n]:
            sure_duration_by_coh.setdefault(coh, ([], []))[0].append(perf.sures[n])
            sure_duration_by_coh[coh][1].append(trial['stimulus'])

    nbins = kwargs.get('nbins', 5)

    sure_by_coh = {}
    for coh, (sure, duration) in sure_duration_by_coh.items():
        Xbins, Ybins, Xedges, _ = partition(np.asarray(duration), np.asarray(sure),
                                            nbins=nbins)
        sure_by_coh[coh] = ((Xedges[:-1] + Xedges[1:])/2,
                             [utils.divide(np.sum(Ybin > 0), len(Ybin))
                              for Ybin in Ybins])

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
    plot.ylim(0, 0.7)

    plot.xlabel('Stimulus duration (ms)')
    plot.ylabel('Probability sure target')

def sort_trials(trialsfile, figspath, name, **kwargs):
    """
    Sort trials by condition.

    """
    # Load trials
    trials, U, Q, Z, A, R, M, perf, states = utils.load(trialsfile)

    #-------------------------------------------------------------------------------------
    # Average within conditions
    #-------------------------------------------------------------------------------------

    # Number of units
    Ntime = states.shape[0]
    N     = states.shape[-1]

    states_by_coh   = {}
    n_states_by_coh = {}
    for n, trial in enumerate(trials):
        coh = trial['left_right']*trial['coh']
        Mn  = np.tile(M[:,n], (N,1)).T

        states_by_coh.setdefault(coh, np.zeros((Ntime, N)))
        states_by_coh[coh] += states[:,n]*Mn
        n_states_by_coh.setdefault(coh, np.zeros((Ntime, N)))
        n_states_by_coh[coh] += Mn
    for coh in states_by_coh:
        for t in xrange(Ntime):
            for i in xrange(N):
                states_by_coh[coh][t,i] = utils.divide(states_by_coh[coh][t,i],
                                                       n_states_by_coh[coh][t,i])

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    lw = kwargs.get('lw', 1.5)

    cohs = np.sort(states_by_coh.keys())
    time = trials[0]['time']

    for unit in xrange(N):
        fig  = Figure()
        plot = fig.add()

        #---------------------------------------------------------------------------------

        for coh in cohs:
            if coh > 0:
                ls = '-'
            else:
                ls = '--'
            plot.plot(time[:-1], states_by_coh[coh][:,unit], ls=ls, lw=lw,
                      color=colors[abs(coh)], label='{}\%'.format(coh))

        plot.xlim(0, max(time))
        plot.ylim(-1, 1)

        #---------------------------------------------------------------------------------

        fig.save(path=figspath, name=name+'_{:03d}'.format(unit))
        fig.close()

def plot_chronometric_function(trialsfile, figspath, name, **kwargs):
    """
    Reaction time as a function of coherence.

    """
    # Load trials
    trials, U, Q, Z, A, R, M, perf, states = utils.load(trialsfile)

    correct_rt_by_coh = {}
    error_rt_by_coh   = {}
    no_decision       = 0
    for n, trial in enumerate(trials):
        coh = trial['coh']
        if coh == 0:
            continue

        if perf.decisions[n]:
            stimulus_start = trial['epoch_durations']['stimulus'][0]
            rt = trial['time'][np.sum(M[:,n])-1] - stimulus_start
            if perf.corrects[n]:
                correct_rt_by_coh.setdefault(coh,[]).append(rt)
            else:
                error_rt_by_coh.setdefault(coh, []).append(rt)
        else:
            no_decision += 1
    print("[ plot_chronometric_function ] {}/{} non-decision trials."
          .format(no_decision, len(trials)))

    min_trials = 0

    # Correct trials
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
            if len(correct_rt_by_coh[coh]) > min_trials:
                correct_idx.append(i)

    # Error trials
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
            if len(error_rt_by_coh[coh]) > min_trials:
                error_idx.append(i)

    print("  Mean RT, correct trials: {:.2f} ms".format(correct_tot/correct_n))
    print("  Mean RT, error trials:   {:.2f} ms".format(error_tot/error_n))

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    fig  = Figure()
    plot = fig.add()

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

    plot.lim('y', [correct_rt, error_rt], lower=0)

    plot.xscale('log')
    plot.xticks([1, 10, 100])
    plot.xticklabels([1, 10, 100])

    plot.xlabel('\% Coherence')
    plot.ylabel('Reaction time (ms)')

    #-------------------------------------------------------------------------------------

    fig.save(path=figspath, name=name)
    fig.close()

def sort_rt_trials(trialsfile, figspath, name, plot=True, **kwargs):
    pass

def sort_postdecision_trials(trialsfile, figspath, name, plot=True, **kwargs):
    """
    Sort postdecision trials by condition.

    """
    # Load trials
    trials, U, Q, Z, A, R, M, perf, states = utils.load(trialsfile)

    # Data shape
    Ntime = states.shape[0]
    N     = states.shape[-1]

    #-------------------------------------------------------------------------------------
    # Aligned to motion onset
    #-------------------------------------------------------------------------------------

    states_by_lr    = {}
    n_states_by_lr  = {}
    states_by_coh   = {}
    n_states_by_coh = {}
    for n, trial in enumerate(trials):
        if trial['wager'] or not perf.decisions[n]:
            continue

        if trial['coh'] == 0:
            continue

        if not perf.corrects[n]:
            continue

        lr  = trial['left_right']
        coh = lr*trial['coh']

        Mn = np.tile(M[:,n], (N,1)).T

        states_by_lr.setdefault(lr, np.zeros((Ntime, N)))
        n_states_by_lr.setdefault(lr, np.zeros((Ntime, N)))

        states_by_coh.setdefault(coh, np.zeros((Ntime, N)))
        n_states_by_coh.setdefault(coh, np.zeros((Ntime, N)))

        states_by_lr[lr] += states[:,n]*Mn
        n_states_by_lr[lr] += Mn

        states_by_coh[coh] += states[:,n]*Mn
        n_states_by_coh[coh] += Mn

    for lr in states_by_lr:
        for t in xrange(Ntime):
            states_by_lr[lr][t] = utils.div(states_by_lr[lr][t], n_states_by_lr[lr][t])

    for coh in states_by_coh:
        for t in xrange(Ntime):
            states_by_coh[coh][t] = utils.div(states_by_coh[coh][t], n_states_by_coh[coh][t])

    #-------------------------------------------------------------------------------------
    # Aligned to saccade
    #-------------------------------------------------------------------------------------

    states_by_coh_saccade   = {}
    n_states_by_coh_saccade = {}
    states_by_lr_saccade    = {}
    n_states_by_lr_saccade  = {}
    for n, trial in enumerate(trials):
        if trial['wager'] or not perf.decisions[n]:
            continue

        if trial['coh'] == 0:
            continue

        if not perf.corrects[n]:
            continue

        lr  = trial['left_right']
        coh = lr*trial['coh']

        states_by_lr_saccade.setdefault(lr, np.zeros((Ntime, N)))
        n_states_by_lr_saccade.setdefault(lr, np.zeros((Ntime, N)))

        states_by_coh_saccade.setdefault(coh, np.zeros((Ntime, N)))
        n_states_by_coh_saccade.setdefault(coh, np.zeros((Ntime, N)))

        t = np.sum(M[:,n])
        Mn = np.tile(M[:t,n], (N,1)).T

        states_by_lr_saccade[lr][-t:] += states[:t,n]*Mn
        n_states_by_lr_saccade[lr][-t:] += Mn

        states_by_coh_saccade[coh][-t:] += states[:t,n]*Mn
        n_states_by_coh_saccade[coh][-t:] += Mn

    for lr in states_by_lr_saccade:
        for t in xrange(Ntime):
            states_by_lr_saccade[lr][t] = utils.div(states_by_lr_saccade[lr][t],
                                                    n_states_by_lr_saccade[lr][t])
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

    for unit in [3]:
        w = 174/25.4
        r = 0.35
        h = r*w
        fig = Figure(w=w, h=h)

        x0 = 0.07
        y0 = 0.1
        w  = 0.15
        h  = 0.8
        dx = 1.3*w
        plots = {
            'motion':  fig.add([x0,    y0, w, h]),
            'saccade': fig.add([x0+dx, y0, w, h])
            }

        #---------------------------------------------------------------------------------

        plot = plots['motion']

        #motion_onset = trials[0]['epoch_durations']['fixation']
        for coh in sorted(states_by_coh.keys()):
            if coh > 0:
                ls = '-'
            else:
                ls = '--'
            plot.plot(time[:-1]-motion_onset, states_by_coh[coh][:,unit], ls=ls, lw=lw,
                      color=colors[abs(coh)], label='{}\%'.format(coh))

        #for lr in sorted(states_by_lr.keys()):
        #    plot.plot(time[:-1], states_by_lr[lr][:,unit], lw=lw,
        #              color=lr_colors[lr], label='{}\%'.format(lr))

        plot.xlim(0, max(time))
        plot.ylim(-1, 1)

        #---------------------------------------------------------------------------------

        plot = plots['saccade']

        for coh in sorted(states_by_coh_saccade.keys()):
            if coh > 0:
                ls = '-'
            else:
                ls = '--'
            plot.plot(time[:-1], states_by_coh_saccade[coh][:,unit], ls=ls, lw=lw,
                      color=colors[abs(coh)], label='{}\%'.format(coh))

        #for lr in sorted(states_by_lr_saccade.keys()):
        #    plot.plot(time[:-1], states_by_lr_saccade[lr][:,unit], lw=lw,
        #              color=lr_colors[lr], label='{}\%'.format(lr))

        plot.xlim(0, max(time))
        plot.ylim(-1, 1)

        #---------------------------------------------------------------------------------

        fig.save(path=figspath, name=name+'_{:03d}'.format(unit))
        fig.close()

#/////////////////////////////////////////////////////////////////////////////////////////

def do(action, args, config):
    """
    Manage tasks.

    """
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    if action == 'trials':
        try:
            n_trials = int(args[0])
        except:
            n_trials = 10

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'])
        rng   = np.random.RandomState(1)

        init, init_b = None, None
        for n in xrange(n_trials):
            init, init_b = plot_trial(pg, model.m, init, init_b, rng, config['figspath'], 'trial_'+str(n))

    elif action in ['trials-psychophysics', 'trials-electrophysiology']:
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

        print("Generating trial conditions ...")
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
            trial   = m.generate_trial_condition(rng, pg.config['dt'], context)

            trials.append(trial)

        print("Running trials ...")
        if action == 'trials-electrophysiology':
            print("Save states.")
            name = 'trials_electrophysiology'
            (U, Q, Z, A, R, M, init, states_0, perf,
             states) = pg.run_trials(trials, return_states=True)
        else:
            print("Behavior only.")
            name = 'trials_psychophysics'
            (U, Q, Z, A, R, M, init, states_0, perf), states = pg.run_trials(trials), None
        perf.display()

        # Save
        trialsfile = os.path.join(config['scratchpath'], name + '.pkl')
        utils.save(trialsfile, (trials, U, Q, Z, A, R, M, perf, states))

        # File size
        size_in_bytes = os.path.getsize(trialsfile)
        print("File size: {:.1f} MB".format(size_in_bytes/2**20))

    elif action == 'trials-save':
        try:
            trials_per_condition = int(args[0])
        except:
            trials_per_condition = 100

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'])
        rng   = np.random.RandomState(1)

        m = config['model'].m

        if 0 in m.cohs:
            cohs = m.cohs[1:]
        else:
            cohs = m.cohs
        left_rights = m.left_rights

        n_conditions = 1 + len(cohs)*len(left_rights)
        n_trials     = n_conditions * trials_per_condition

        init = None

        trials = []

        backspace = 0
        for n in xrange(n_trials):
            s = '{}/{}'.format(n+1, n_trials)
            utils.println(backspace*'\b' + s)
            backspace = len(s)

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
            trial   = m.generate_trial_condition(rng, context)

            trials.append(trial)
        print("")

        (U, Q, Z, A, R, M, init, states_0, perf,
         states) = pg.run_trials(trials, init=init, return_states=True)
        perf.display()

        # Save
        trialsfile = os.path.join(config['scratchpath'], 'trials.pkl')
        utils.save(trialsfile, (trials, U, Q, Z, A, R, M, perf, states))

        # File size
        size_in_bytes = os.path.getsize(trialsfile)
        print("File size: {:.1f} MB".format(size_in_bytes/2**20))

    elif action == 'stimulus-duration':
        trialsfile = os.path.join(config['scratchpath'], 'trials_psychophysics.pkl')
        plot_stimulus_duration(trialsfile, config['figspath'], 'stimulus_duration',
                               nbins=10)

    elif action == 'stimulus-duration-wager':
        trialsfile = os.path.join(config['scratchpath'], 'trials_psychophysics.pkl')

        fig  = Figure(axislabelsize=10, ticklabelsize=9)
        plot = fig.add()

        plot_stimulus_duration_wager(trialsfile, plot, nbins=7)

        fig.save(path=config['figspath'], name='stimulus_duration')
        fig.close()

    elif action == 'sure-stimulus-duration':
        trialsfile = os.path.join(config['scratchpath'], 'trials_psychophysics.pkl')

        fig  = Figure(axislabelsize=10, ticklabelsize=9)
        plot = fig.add()

        plot_sure_stimulus_duration(trialsfile, plot, nbins=7)

        fig.save(path=config['figspath'], name='sure_stimulus_duration')
        fig.close()

    elif action == 'chronometric':
        trialsfile = os.path.join(config['scratchpath'], 'trials_psychophysics.pkl')
        plot_chronometric_function(trialsfile, config['figspath'], 'chronometric')

    elif action == 'psychometric':
        trialsfile = os.path.join(config['scratchpath'], 'trials_psychophysics.pkl')
        psychometric(trialsfile, config['model'].m, config['figspath'], 'psychometric')

    elif action == 'sort-trials':
        trialsfile = os.path.join(config['scratchpath'], 'trials_electrophysiology.pkl')
        sort_trials(trialsfile, config['figspath'], 'sorted')

    elif action == 'sort-postdecision-trials':
        trialsfile = os.path.join(config['scratchpath'], 'trials_electrophysiology.pkl')
        sort_postdecision_trials(trialsfile, config['figspath'], 'sorted')

    elif action == 'psychometric-only':
        try:
            n_trials = int(args[0])
        except:
            n_trials = 1200

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'])
        rng   = np.random.RandomState(1)
        plot_psychometric(pg, rng, model.m, n_trials, config['figspath'], 'psychometric')

    elif action == 'info':
        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'])
