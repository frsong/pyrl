from __future__ import absolute_import, division

import os

import numpy as np

from pyrl          import fittools, runtools, tasktools, utils
from pyrl.figtools import apply_alpha, Figure

#/////////////////////////////////////////////////////////////////////////////////////////

def plot_trial(pg, m, init, init_b, rng, figspath, name):
    context = {}
    if 0 not in m.cohs:
        context['cohs'] = [0] + m.cohs
    trial = m.generate_trial_condition(rng, context)

    U, Z, A, R, M, init, states_0, perf = pg.run_trials([trial], init=init)
    if pg.baseline_net is not None:
        (init_b, baseline_states_0, b,
         rpe) = pg.baseline_run_trials(U, A, R, M, init=init_b)
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
    plot.ylim(m.R_TERMINATE, m.R_CORRECT)
    plot.xlabel('Time (ms)')
    plot.ylabel('Reward')

    plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------

    fig.save(path=figspath, name=name)
    fig.close()

    #-------------------------------------------------------------------------------------

    return init, init_b

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

        if perf.choices[n] == 'L':
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

def get_preferred_targets(trials, perf, r, verbose=False):
    """
    Determine preferred targets.

    """
    dprime = compute_dprime(trials, perf, r)
    if verbose:
        for i in xrange(len(dprime)):
            if abs(dprime[i]) > 0.5:
                print(i, dprime[i])

    return 2*(dprime > 0) - 1

def sort_func(s, preferred_targets, target, trial):
    choices = preferred_targets*target

    if s == 'choice':
        return [(choice,) for choice in choices]
    elif s == 'motion-choice':
        cohs = preferred_targets*trial['left_right_m']*trial['coh_m']
        return [(choice, coh, trial['context']) for choice, coh in zip(choices, cohs)]
    elif s == 'color-choice':
        cohs = preferred_targets*trial['left_right_c']*trial['coh_c']
        return [(choice, coh, trial['context']) for choice, coh in zip(choices, cohs)]
    elif s == 'context-choice':
        return [(choice, trial['context']) for choice in choices]
    elif s == 'all':
        cohs_m = preferred_targets*trial['left_right_m']*trial['coh_m']
        cohs_c = preferred_targets*trial['left_right_c']*trial['coh_c']
        return [(choice, coh_m, coh_c, trial['context'])
                for choice, coh_m, coh_c in zip(choices, cohs_m, cohs_c)]
    else:
        raise ValueError

def sort(trialsfile, all_plots, units=None, network='p', **kwargs):
    """
    Sort trials.

    """
    # Load trials
    trials, U, Z, Z_b, A, P, M, perf, r_p, r_v = utils.load(trialsfile)

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
    # Preferred targets
    #=====================================================================================

    preferred_targets = get_preferred_targets(trials, perf, r)

    #=====================================================================================
    # Sort trials
    #=====================================================================================

    sortby = ['choice', 'motion-choice', 'color-choice', 'context-choice', 'all']

    #-------------------------------------------------------------------------------------
    # Sort
    #-------------------------------------------------------------------------------------

    sorted_trials = {s: {} for s in sortby}
    X  = 0
    X2 = 0
    NX = 0
    for n, trial in enumerate(trials):
        if perf.choices[n] == 'R':
            target = +1
        else:
            target = -1

        if perf.corrects[n]:
            for s in sortby:
                sorted_trial = sort_func(s, preferred_targets, target, trial)
                for u, cond in enumerate(sorted_trial):
                    sorted_trials[s].setdefault(cond, []).append((n, u))

        # For normalizing
        Mn  = np.tile(M[:,n], (N,1)).T
        Rn  = r[:,n]*Mn
        X  += np.sum(Rn)
        X2 += np.sum(Rn**2)
        NX += np.sum(Mn)
    mean = X/NX
    sd   = np.sqrt(X2/NX - mean**2)

    #-------------------------------------------------------------------------------------
    # Average within conditions
    #-------------------------------------------------------------------------------------

    for s in sorted_trials:
        # Collect
        trials_by_cond = {}
        for cond, n_u in sorted_trials[s].items():
            # Storage
            trials_by_cond.setdefault(cond, {'r': np.zeros((Ntime_a, N)),
                                             'n': np.zeros((Ntime_a, N))})
            for n, u in n_u:
                # Firing rates
                Mn  = M[:,n]
                Rnu = r[:,n,u]*Mn

                # Normalize
                Rnu = (Rnu - mean)/sd

                # Align point
                t0 = trials[n]['epochs']['stimulus'][0] - 1

                # Before
                n_b = Rnu[:t0].shape[0]
                trials_by_cond[cond]['r'][Ntime-1-n_b:Ntime-1,u] += Rnu[:t0]
                trials_by_cond[cond]['n'][Ntime-1-n_b:Ntime-1,u] += Mn[:t0]

                # After
                n_a = Rnu[t0:].shape[0]
                trials_by_cond[cond]['r'][Ntime-1:Ntime-1+n_a,u] += Rnu[t0:]
                trials_by_cond[cond]['n'][Ntime-1:Ntime-1+n_a,u] += Mn[t0:]

        # Average
        for cond in trials_by_cond:
            trials_by_cond[cond] = utils.div(trials_by_cond[cond]['r'],
                                             trials_by_cond[cond]['n'])

        # Save
        sorted_trials[s] = trials_by_cond

    if all_plots is None:
        return time_a, sorted_trials

    #=====================================================================================
    # Plot functions
    #=====================================================================================

    lw = kwargs.get('lw', 1)

    linestyles = {
        +1: '-',
        -1: '--'
        }

    def plot_choice(plot, unit, w):
        t = time_a[w]
        y = [[0, 0.5]]
        for (choice,), r_cond in sorted_trials['choice'].items():
            plot.plot(t, r_cond[w,unit], linestyles[choice], color=Figure.colors('red'), lw=lw)
            y.append(r_cond[w,unit])
        plot.lim('y', y)

        return t, y

    def plot_motion_choice(plot, unit, w):
        cohs = []
        for (choice, signed_coh, context) in sorted_trials['motion-choice']:
            cohs.append(abs(signed_coh))
        cohs = sorted(list(set(cohs)))

        t = time_a[w]
        y = [[0, 0.5]]
        for (choice, signed_coh, context), r_cond in sorted_trials['motion-choice'].items():
            if context != 'm':
                continue

            idx = cohs.index(abs(signed_coh))
            basecolor = 'k'
            if idx == 0:
                color = apply_alpha(basecolor, 0.4)
            elif idx == 1:
                color = apply_alpha(basecolor, 0.7)
            else:
                color = apply_alpha(basecolor, 1)

            plot.plot(t, r_cond[w,unit], linestyles[choice], color=color, lw=lw)
            y.append(r_cond[w,unit])
        plot.lim('y', y)

        return t, y

    def plot_color_choice(plot, unit, w):
        cohs = []
        for (choice, signed_coh, context) in sorted_trials['color-choice']:
            cohs.append(abs(signed_coh))
        cohs = sorted(list(set(cohs)))

        t = time_a[w]
        y = [[0, 0.5]]
        for (choice, signed_coh, context), r_cond in sorted_trials['color-choice'].items():
            if context != 'c':
                continue

            idx = cohs.index(abs(signed_coh))
            basecolor = Figure.colors('darkblue')
            if idx == 0:
                color = apply_alpha(basecolor, 0.4)
            elif idx == 1:
                color = apply_alpha(basecolor, 0.7)
            else:
                color = apply_alpha(basecolor, 1)

            plot.plot(t, r_cond[w,unit], linestyles[choice], color=color, lw=lw)
            y.append(r_cond[w,unit])
        plot.lim('y', y)

        return t, y

    def plot_context_choice(plot, unit, w):
        t = time_a[w]
        y = [[0, 0.5]]
        for (choice, context), r_cond in sorted_trials['context-choice'].items():
            if context == 'm':
                color = 'k'
            else:
                color = Figure.colors('darkblue')

            plot.plot(t, r_cond[w,unit], linestyles[choice], color=color, lw=lw)
            y.append(r_cond[w, unit])
        plot.lim('y', y)

        return t, y

    #=====================================================================================
    # Plot
    #=====================================================================================

    if units is not None:
        tmin = kwargs.get('tmin', 100)
        tmax = kwargs.get('tmax', 850)
        w, = np.where((tmin <= time_a ) & (time_a <= tmax))

        for plots, unit in zip(all_plots, units):
            yall = []

            plot = plots['choice']
            t, y = plot_choice(plot, unit, w)
            yall += y

            plot = plots['motion-choice']
            t, y = plot_motion_choice(plot, unit, w)
            yall += y

            plot = plots['color-choice']
            t, y = plot_color_choice(plot, unit, w)
            yall += y

            plot = plots['context-choice']
            t, y = plot_context_choice(plot, unit, w)
            yall += y
    else:
        figspath, name = all_plots
        for unit in xrange(N):
            w   = 2.5
            h   = 6
            fig = Figure(w=w, h=h, axislabelsize=7.5, ticklabelsize=6.5)

            w  = 0.55
            h  = 0.17
            x0 = 0.3
            y0 = 0.77
            dy = 0.06

            fig.add('choice',         [x0, y0, w, h])
            fig.add('motion-choice',  [x0, fig['choice'].y-dy-h, w, h])
            fig.add('color-choice',   [x0, fig['motion-choice'].y-dy-h, w, h])
            fig.add('context-choice', [x0, fig['color-choice'].y-dy-h, w, h])

            #-----------------------------------------------------------------------------

            w, = np.where((-100 <= time_a ) & (time_a <= 750))

            yall = []

            plot = fig['choice']
            t, y = plot_choice(plot, unit, w)
            yall += y

            plot = fig['motion-choice']
            t, y = plot_motion_choice(plot, unit, w)
            yall += y

            plot = fig['color-choice']
            t, y = plot_color_choice(plot, unit, w)
            yall += y

            plot = fig['context-choice']
            t, y = plot_context_choice(plot, unit, w)
            yall += y

            for plot in fig.plots.values():
                plot.lim('y', yall)

            #-----------------------------------------------------------------------------

            fig.save(path=figspath, name=name+'_{}{:03d}'.format(network, unit))
            fig.close()

#/////////////////////////////////////////////////////////////////////////////////////////

def is_active(r):
    return np.std(r, axis=0) > 0.1

def get_active_units(r, M):
    N   = r.shape[-1]
    M_  = (np.tile(M.T, (N, 1, 1))).T
    r_  = r*M_
    n   = np.sum(M)
    var = (r_**2).sum(axis=0).sum(axis=0)/n - (r_.sum(axis=0).sum(axis=0)/n)**2

    return np.where(np.sqrt(var) > 0.2)[0]

# Regression coefficients
CHOICE         = 0
MOTION         = 1
COLOUR         = 2
CONTEXT        = 3
CONSTANT       = 4
CHOICE_MOTION  = 5
CHOICE_COLOUR  = 6
CHOICE_CONTEXT = 7
MOTION_COLOUR  = 8
MOTION_CONTEXT = 9
COLOUR_CONTEXT = 10
nreg           = 11

def plot_taskaxes(plot, yax, p_vc, basecolor):
    abscohs = []
    for choice, coh, context in p_vc:
        abscohs.append(abs(coh))
    abscohs = sorted(list(set(abscohs)))

    #-------------------------------------------------------------------------------------
    # Subtract mean
    #-------------------------------------------------------------------------------------

    p = p_vc.values()[0]
    Xchoice = np.zeros_like(p[CHOICE])
    Xmotion = np.zeros_like(p[MOTION])
    Xcolour = np.zeros_like(p[COLOUR])

    for p in p_vc.values():
        Xchoice += p[CHOICE]
        Xmotion += p[MOTION]
        Xcolour += p[COLOUR]
    mean_choice = Xchoice/len(p_vc)
    mean_motion = Xmotion/len(p_vc)
    mean_colour = Xcolour/len(p_vc)

    for cond, p in p_vc.items():
        p[CHOICE] -= mean_choice
        p[MOTION] -= mean_motion
        p[COLOUR] -= mean_colour

    #-------------------------------------------------------------------------------------

    xall = []
    yall = []
    for cond, p in p_vc.items():
        idx = abscohs.index(abs(cond[1]))
        if idx == 0:
            color = apply_alpha(basecolor, 0.4)
        elif idx == 1:
            color = apply_alpha(basecolor, 0.7)
        else:
            color = apply_alpha(basecolor, 1)

        if cond[1] > 0:
            prop = dict(mfc=color, mec=color, ms=2.5, mew=0.5)
        else:
            prop = dict(mfc='w', mec=color, ms=2.5, mew=0.5)

        plot.plot(p[CHOICE],      p[yax],      '-', color=color, lw=0.75)
        plot.plot(p[CHOICE][::2], p[yax][::2], 'o', color=color, **prop)

        xall.append(p[CHOICE])
        yall.append(p[yax])

    if yax == MOTION:
        plot.ylabel('Motion')
    elif yax == COLOUR:
        plot.ylabel('Color')

    return np.concatenate(xall), np.concatenate(yall)

def plot_statespace(units, t, sorted_trials, Q, plots):
    # Task axes
    M = Q.T

    # Epoch to plot
    w, = np.where((0 <= t) & (t <= 800))

    # Down-sample
    dt   = t[1] - t[0]
    step = int(50/dt)
    w    = w[::step]

    # Colors
    color_m = 'k'
    color_c = Figure.colors('darkblue')

    xall = []
    yall = []

    #-------------------------------------------------------------------------------------
    # Labels
    #-------------------------------------------------------------------------------------

    plots['c1'].xlabel('Choice')

    #-------------------------------------------------------------------------------------
    # Motion context: motion vs. choice, sorted by coherence
    #-------------------------------------------------------------------------------------

    plot = plots['m1']

    p_vc = {}
    for cond, r in sorted_trials['motion-choice'].items():
        if cond[2] == 'm':
            p_vc[cond] = M.dot(r.T[units,:][:,w])
    x, y = plot_taskaxes(plot, MOTION, p_vc, color_m)
    xall.append(x)
    yall.append(y)

    plot.ylabel('Motion')

    #-------------------------------------------------------------------------------------
    # Motion context: motion vs. choice, sorted by coherence
    #-------------------------------------------------------------------------------------

    plot = plots['m2']
    p_vc = {}
    for cond, r in sorted_trials['motion-choice'].items():
        if cond[2] == 'm':
            p_vc[cond] = M.dot(r.T[units,:][:,w])
    x, y = plot_taskaxes(plot, COLOUR, p_vc, color_m)
    xall.append(x)
    yall.append(y)

    #-------------------------------------------------------------------------------------
    # Motion context: colour vs. choice, sorted by colour
    #-------------------------------------------------------------------------------------

    plot = plots['m3']
    p_vc = {}
    for cond, r in sorted_trials['color-choice'].items():
        if cond[2] == 'm':
            p_vc[cond] = M.dot(r.T[units,:][:,w])
    x, y = plot_taskaxes(plot, COLOUR, p_vc, color_c)
    xall.append(x)
    yall.append(y)

    #-------------------------------------------------------------------------------------
    # Colour context: motion vs. choice, sorted by motion
    #-------------------------------------------------------------------------------------

    plot = plots['c1']
    p_vc = {}
    for cond, r in sorted_trials['motion-choice'].items():
        if cond[2] == 'c':
            p_vc[cond] = M.dot(r.T[units,:][:,w])
    x, y = plot_taskaxes(plot, MOTION, p_vc, color_m)
    xall.append(x)
    yall.append(y)

    #-------------------------------------------------------------------------------------
    # Colour context: motion vs. choice, sorted by colour
    #-------------------------------------------------------------------------------------

    plot = plots['c2']
    p_vc = {}
    for cond, r in sorted_trials['color-choice'].items():
        if cond[2] == 'c':
            p_vc[cond] = M.dot(r.T[units,:][:,w])
    x, y = plot_taskaxes(plot, MOTION, p_vc, color_c)
    xall.append(x)
    yall.append(y)

    #-------------------------------------------------------------------------------------
    # Colour context: colour vs. choice, sorted by colour
    #-------------------------------------------------------------------------------------

    plot = plots['c3']
    p_vc = {}
    for cond, r in sorted_trials['color-choice'].items():
        if cond[2] == 'c':
            p_vc[cond] = M.dot(r.T[units,:][:,w])
    x, y = plot_taskaxes(plot, COLOUR, p_vc, color_c)
    xall.append(x)
    yall.append(y)

    #-------------------------------------------------------------------------------------
    # Shared axes
    #-------------------------------------------------------------------------------------

    xall = np.concatenate(xall)
    yall = np.concatenate(yall)

    for plot in plots.values():
        #plot.aspect(1.5)
        plot.lim('x', xall)
        plot.lim('y', yall)

def statespace(trialsfile, plots=None, dt_reg=50, **kwargs):
    """
    State-space analysis.

    """
    # Load trials
    trials_, U, Z, Z_b, A, P, M, perf, r_p, r_v = utils.load(trialsfile)

    # Use policy network for this analysis
    r = r_p
    N = r.shape[-1]

    # Time step
    time  = trials_[0]['time']
    Ntime = len(time)
    dt    = time[1] - time[0]
    step  = int(dt_reg/dt)

    #=====================================================================================
    # Setup
    #=====================================================================================

    # Active units
    units = get_active_units(r, M)
    print("[ mante.statespace ] Performing regression on {} active units."
          .format(len(units)))

    # Preferred targets for active units
    preferred_targets = get_preferred_targets(trials_, perf, r)[units]

    # Stimulus period
    stimulus = np.asarray(trials_[0]['epochs']['stimulus'])[::step]

    trials = []
    cohs_m = []
    cohs_c = []
    for n, trial_ in enumerate(trials_):
        if perf.choices[n] is None:
            continue

        cohs_m.append(trial_['coh_m'])
        cohs_c.append(trial_['coh_c'])

        trial           = {}
        trial['target'] = +1 if perf.choices[n] == 'R' else -1
        trial['t']      = time[stimulus]
        trial['r']      = r[stimulus,n,:][:,units].T
        trials.append(trial)
    maxcoh_m = max(cohs_m)
    maxcoh_c = max(cohs_c)

    #-------------------------------------------------------------------------------------
    # Normalize
    #-------------------------------------------------------------------------------------

    X  = 0
    X2 = 0
    n  = 0
    for trial in trials:
        r   = trial['r']
        X  += np.sum(r,    axis=1)
        X2 += np.sum(r**2, axis=1)
        n  += r.shape[1]
    mean = X/n
    sd   = np.sqrt(X2/n - mean**2)

    mean = np.tile(mean, (r.shape[1], 1)).T
    sd   = np.tile(sd,   (r.shape[1], 1)).T
    for trial in trials:
        trial['r'] = (trial['r'] - mean)/sd

    #-------------------------------------------------------------------------------------
    # Regress
    #-------------------------------------------------------------------------------------

    nunits, ntime = trials[0]['r'].shape
    ntrials = len(trials)

    # Coefficient matrix
    r = np.zeros((nunits, ntime, ntrials))
    F = np.zeros((nunits, nreg, ntrials))
    for i, trial in enumerate(trials):
        info = trials_[i]

        # First-order terms
        r[:,:,i]       = trial['r']
        F[:,CHOICE,i]  = preferred_targets*trial['target']
        F[:,MOTION,i]  = preferred_targets*info['left_right_m']*info['coh_m']/maxcoh_m
        F[:,COLOUR,i]  = preferred_targets*info['left_right_c']*info['coh_c']/maxcoh_c
        F[:,CONTEXT,i] = +1 if info['context'] == 'm' else -1

        # Interaction terms
        F[:,CHOICE_MOTION, i] = F[:,CHOICE,i]*F[:,MOTION,i]
        F[:,CHOICE_COLOUR, i] = F[:,CHOICE,i]*F[:,COLOUR,i]
        F[:,CHOICE_CONTEXT,i] = F[:,CHOICE,i]*F[:,CONTEXT,i]
        F[:,MOTION_COLOUR, i] = F[:,MOTION,i]*F[:,COLOUR,i]
        F[:,MOTION_CONTEXT,i] = F[:,MOTION,i]*F[:,CONTEXT,i]
        F[:,COLOUR_CONTEXT,i] = F[:,COLOUR,i]*F[:,CONTEXT,i]
    F[:,CONSTANT,:] = 1

    # Regression coefficients
    beta = np.zeros((nunits, ntime, nreg))
    for i in xrange(nunits):
        A = np.linalg.inv(F[i].dot(F[i].T)).dot(F[i])
        for k in xrange(ntime):
            beta[i,k] = A.dot(r[i,k])
            if np.any(np.isnan(beta[i,k])):
                raise RuntimeError("[ mante.regress ] Regression failed.")

    #-------------------------------------------------------------------------------------
    # Sort trials
    #-------------------------------------------------------------------------------------

    utils.println("[ mante.statespace ] Sorting trials ...")
    time_a, sorted_trials = sort(trialsfile, None)
    print(" done!")

    #-------------------------------------------------------------------------------------
    # Denoising matrix
    #-------------------------------------------------------------------------------------

    '''
    all_conditions = sorted_trials['all']
    for cond, r in all_conditions.items():
        all_conditions[cond] = r.T[units,::step]

    # Data matrix
    X = np.zeros((all_conditions.values()[0].shape[0],
                  len(all_conditions)*all_conditions.values()[0].shape[1]))
    c = 0
    for cond, r in sorted_trials['all'].items():
        X[:,c:c+r.shape[1]] = r
        c += r.shape[1]

    U, S, V = np.linalg.svd(X.T)
    assert np.all(S[:-1] >= S[1:])

    npca = 12
    W    = V[:npca,:]
    D    = (W.T).dot(W)
    assert np.all(D.T == D)
    '''

    #-------------------------------------------------------------------------------------
    # Task axes
    #-------------------------------------------------------------------------------------

    # Rearrange from (units, time, reg) to (reg, time, units)
    beta = np.swapaxes(beta, 0, 2)

    # Denoise
    # beta = beta.dot(D.T)

    # Time-independent regression vectors
    beta_max = np.zeros((nreg, nunits))
    for v in xrange(nreg):
        imax        = np.argmax(np.linalg.norm(beta[v], axis=1))
        beta_max[v] = beta[v,imax]

    Bmax = beta_max[:4].T
    Q, R = np.linalg.qr(Bmax)
    Q    = Q*np.sign(np.diag(R))

    #=====================================================================================

    if isinstance(plots, dict):
        plot_statespace(units, time_a, sorted_trials, Q, plots)
    else:
        figspath, name = plots

        w = utils.mm_to_inch(174)
        r = 0.65
        fig = Figure(w=w, r=r, thickness=0.8, axislabelsize=9, ticklabelsize=7,
                     labelpadx=4.5, labelpady=4.5)

        w  = 0.24
        h  = 0.35
        x0 = 0.1
        y0 = 0.11
        dx = 0.07
        dy = 0.15

        fig.add('c1', [x0, y0, w, h])
        fig.add('c2', [fig[-1].right+dx, fig[-1].y, w, h])
        fig.add('c3', [fig[-1].right+dx, fig[-1].y, w, h])
        fig.add('m1', [fig['c1'].x, fig['c1'].top+dy, w, h])
        fig.add('m2', [fig[-1].right+dx, fig[-1].y, w, h])
        fig.add('m3', [fig[-1].right+dx, fig[-1].y, w, h])

        #---------------------------------------------------------------------------------

        plot_statespace(units, time_a, sorted_trials, Q, fig.plots)

        #---------------------------------------------------------------------------------
        # Legend
        #---------------------------------------------------------------------------------

        ms_filled = 2.5
        ms_empty  = 2.5

        mew_filled = 0.5
        mew_empty  = 0.5

        y  = 1.2
        dx = 0.08
        dy = 0.06

        fontsize = 4.75

        for context, plot, basecolor in zip(['Motion', 'Color'],
                                            [fig['c1'], fig['c3']],
                                            ['k', Figure.colors('darkblue')]):
            transform = plot.ax.transAxes
            colors    = [apply_alpha(basecolor, alpha) for alpha in [0.4, 0.7, 1]]
            for i in xrange(3):
                plot.plot(0.5+(i+0.5)*dx, y, 'o', mfc=colors[i], mec=colors[i],
                          ms=ms_filled, mew=mew_filled, transform=transform)
                plot.plot(0.5-(i+0.5)*dx, y, 'o', mfc='none', mec=colors[i],
                          ms=ms_empty, mew=mew_empty, transform=transform)

            # Strength label
            plot.text(0.5, y+dy, 'Weak', ha='center', va='bottom', fontsize=fontsize,
                      color=colors[0], transform=transform)
            plot.text(0.5+2.5*dx, y+dy, 'Strong', ha='center', va='bottom',
                      fontsize=fontsize, color=colors[-1], transform=transform)
            plot.text(0.5-2.5*dx, y+dy, 'Strong', ha='center', va='bottom',
                      fontsize=fontsize, color=colors[-1], transform=transform)

            if context == 'Motion':
                plot.text(0.5-5*dx, y, context, ha='right', va='center',
                          fontsize=1.2*fontsize, color=colors[-1], transform=transform)
            else:
                plot.text(0.5+5*dx, y, context, ha='left', va='center',
                          fontsize=1.2*fontsize, color=colors[-1], transform=transform)

            # Choice label
            plot.text(0.5+2.5*dx, y-dy, 'To choice 1', ha='center', va='top',
                      fontsize=fontsize, color='k', transform=transform)
            plot.text(0.5-2.5*dx, y-dy, 'To choice 2', ha='center', va='top',
                      fontsize=fontsize, color='k', transform=transform)

        #-----------------------------------------------------------------------------

        fig.save(path=figspath, name=name)
        fig.close()

#/////////////////////////////////////////////////////////////////////////////////////////

def psychometric(trialsfile, plots=None, **kwargs):
    """
    Compute and plot the psychometric functions.

    """
    # Load trials
    trials, A, R, M, perf = utils.load(trialsfile)

    # Sort results by context, coherence
    results = {cond: {} for cond in ['mm', 'mc', 'cm', 'cc']}
    for n, trial in enumerate(trials):
        if not perf.decisions[n]:
            continue

        coh_m = trial['left_right_m']*trial['coh_m']
        coh_c = trial['left_right_c']*trial['coh_c']

        if perf.choices[n] == 'R':
            choice = 1
        else:
            choice = 0

        if trial['context'] == 'm':
            motion_choices = results['mm'].setdefault(coh_m, [])
            color_choices  = results['mc'].setdefault(coh_c, [])
        else:
            motion_choices = results['cm'].setdefault(coh_m, [])
            color_choices  = results['cc'].setdefault(coh_c, [])
        motion_choices.append(choice)
        color_choices.append(choice)

    # Convert to P(right)
    for k, choices_by_coh in results.items():
        cohs = np.sort(choices_by_coh.keys())
        p1   = np.zeros(len(cohs))
        for i, coh in enumerate(cohs):
            choices = choices_by_coh[coh]
            p1[i]   = sum(choices)/len(choices)
        results[k] = (cohs, p1)

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    if plots is not None:
        lw      = kwargs.get('lw', 1.25)
        ms      = kwargs.get('ms', 5)
        color_m = 'k'
        color_c = Figure.colors('darkblue')

        x_all = {'m': [], 'c': []}
        for k, v in results.items():
            # Context
            if k[0] == 'm':
                color = color_m
                label = 'Motion context'
            else:
                color = color_c
                label = 'Color context'

            plot = plots[k[1]]

            cohs, p1 = v
            plot.plot(cohs, 100*p1, 'o', ms=ms, mew=0, mfc=color, zorder=10)
            props = dict(color=color, lw=lw, zorder=5, label=label)
            try:
                popt, func = fittools.fit_psychometric(cohs, p1)

                fit_cohs = np.linspace(min(cohs), max(cohs), 201)
                fit_p1   = func(fit_cohs, **popt)
                plot.plot(fit_cohs, 100*fit_p1, **props)
            except RuntimeError:
                print("Unable to fit, drawing a line through the points.")
                plot.plot(cohs, 100*p1, **props)
            x_all[k[1]].append(cohs)

        for s in ['m', 'c']:
            plots[s].lim('x', x_all[s])
            plots[s].ylim(0, 100)
            plots[s].yticks([0, 50, 100])

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

#/////////////////////////////////////////////////////////////////////////////////////////

def do(action, args, config):
    """
    Manage tasks.

    """
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    #=====================================================================================

    if action == 'performance':
        fig  = Figure(axislabelsize=10, ticklabelsize=9)
        plot = fig.add()

        performance(config['savefile'], plot)

        fig.save(path=config['figspath'], name='performance')
        fig.close()

    #=====================================================================================

    elif 'trials' in action:
        try:
            trials_per_condition = int(args[0])
        except:
            trials_per_condition = 1000

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'], config['dt'])

        spec         = model.spec
        mcs          = spec.contexts
        cohs         = spec.cohs
        left_rights  = spec.left_rights
        n_conditions = spec.n_conditions
        n_trials     = n_conditions * trials_per_condition

        print("{} trials".format(n_trials))
        task   = model.Task()
        trials = []
        for n in xrange(n_trials):
            k = tasktools.unravel_index(n, (len(mcs),
                                            len(left_rights), len(left_rights),
                                            len(cohs), len(cohs)))
            context = {
                'context':      mcs[k.pop(0)],
                'left_right_m': left_rights[k.pop(0)],
                'left_right_c': left_rights[k.pop(0)],
                'coh_m':        cohs[k.pop(0)],
                'coh_c':        cohs[k.pop(0)]
                }
            trials.append(task.get_condition(pg.rng, pg.dt, context))
        runtools.run(action, trials, pg, config['trialspath'])

    #=====================================================================================

    elif action == 'psychometric':
        trialsfile = runtools.behaviorfile(config['trialspath'])

        fig = Figure(w=6, h=2.7)

        x0 = 0.12
        y0 = 0.2
        w  = 0.36
        h  = 0.7
        dx = 0.1

        fig.add('m', [x0, y0, w, h])
        fig.add('c', [fig[-1].right+dx, y0, w, h])

        psychometric(trialsfile, fig.plots)

        fig['m'].xlabel('Percent motion coherence')
        fig['m'].ylabel('Percent right')
        fig['c'].xlabel('Percent color coherence')

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

    #=====================================================================================

    elif action == 'statespace':
        trialsfile = runtools.activityfile(config['trialspath'])
        statespace(trialsfile, (config['figspath'], 'statespace'))
