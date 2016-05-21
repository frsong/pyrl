from __future__ import absolute_import, division

import os

import numpy as np

from pyrl                import tasktools
from pyrl                import utils
from pyrl.policygradient import PolicyGradient

from pycog.figtools  import Figure

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

def plot_psychometric(pg, rng, m, n_trials, figspath, name, **kwargs):
    #-------------------------------------------------------------------------------------
    # Run trials
    #-------------------------------------------------------------------------------------

    (U, Z, A, R, M, init_, states_0_,
     perf) = pg.run_trials(n_trials, contextupdater=m.ContextUpdater(None, verbose=True))

    nD = 0
    nL = 0
    nR = 0
    for n in xrange(n_trials):
        t = np.sum(M[:,n])
        a = int(np.argmax(A[t-1,n]))
        if a in [m.actions['SACCADE_LEFT'], m.actions['SACCADE_RIGHT']]:
            nD += 1
            if a == m.actions['SACCADE_LEFT']:
                nL += 1
            else:
                nR += 1

    perf.display()
    print(nL, nR, nD)

    return

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

        U, Z, A, R, M, init, states_0, perf = pg.run_trials(trials, init=init)
        perf.display()

        # Save
        trialsfile = os.path.join(config['scratchpath'], 'trials.pkl')
        utils.save(trialsfile, (trials, U, Z, A, R, M, perf))

    elif action == 'psychometric':
        try:
            n_trials = int(args[0])
        except:
            n_trials = 1100

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'])
        rng   = np.random.RandomState(1)
        plot_psychometric(pg, rng, model.m, n_trials, config['figspath'], 'psychometric')
