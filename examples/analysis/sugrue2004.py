from __future__ import absolute_import, division

import os

import numpy as np

from scipy.ndimage.filters import gaussian_filter1d

from pyrl          import datatools, fittools, tasktools, utils
from pyrl.figtools import Figure, mpl

#/////////////////////////////////////////////////////////////////////////////////////////

THIS = 'pyrl.analysis.sugrue2004'

#/////////////////////////////////////////////////////////////////////////////////////////

def cumulative_choice(savefile, plot, **kwargs):
    perf = utils.load(savefile)['perf_history'][0]

    lw = kwargs.get('lw', 1.5)

    plot.equal()

    R     = perf.choiceR
    G     = perf.choiceG
    rateR = perf.rateR
    rateG = perf.rateG
    rewards = perf.rewards
    #rewardR = perf.rewardR
    #rewardG = perf.rewardG

    last_n = 850
    last_n = 2000
    if len(R) > last_n:
        R = R[-last_n:]
        G = G[-last_n:]
        rateR = rateR[-last_n:]
        rateG = rateG[-last_n:]
        rewards = rewards[-last_n:]
        #rewardR = rewardR[-last_n:]
        #rewardG = rewardG[-last_n:]

    print("R      ", R[:20])
    #print("rR", rewardR[:20])
    print("G      ", G[:20])
    #print("rG", rewardG[:20])
    print("rewards", rewards[:20])

    rateR = np.asarray(rateR)
    rateG = np.asarray(rateG)

    x = np.cumsum(G)
    y = np.cumsum(R)
    plot.plot(x, y, '-', color=Figure.colors('blue'), lw=lw, zorder=10)

    r   = rateR/(rateR + rateG)
    tot = np.ones_like(r)
    xx = np.cumsum(r*tot)
    yy = np.cumsum((1-r)*tot)
    plot.plot(xx, yy, '-', color='k', lw=lw, zorder=5)

    lim = max([x[-1], y[-1], xx[-1], yy[-1]])
    plot.xlim(0, lim)
    plot.ylim(0, lim)

def instantaneous_choice(savefile, plot, **kwargs):
    perf = utils.load(savefile)['perf_history'][0]

    lw = kwargs.get('lw', 1.25)

    R_       = perf.choiceR
    G_       = perf.choiceG
    rateR_   = perf.rateR
    rateG_   = perf.rateG
    rewardR_ = perf.rewardR
    rewardG_ = perf.rewardG
    rewards_ = perf.rewards

    R       = []
    G       = []
    rateR   = []
    rateG   = []
    rewardR = []
    rewardG = []
    rewards = []

    choice_tm1 = None
    for i in xrange(len(R_)):
        if choice_tm1 is not None and i < len(R_)-1:
            if R_[i] == 1:
                if choice_tm1 == 'G' and R_[i+1] == 1:
                    continue
            elif G_[i] == 1:
                if choice_tm1 == 'R' and G_[i+1] == 1:
                    continue
            else:
                raise

        R.append(R_[i])
        G.append(G_[i])
        rateR.append(rateR_[i])
        rateG.append(rateG_[i])
        rewardR.append(rewardR_[i])
        rewardG.append(rewardG_[i])
        rewards.append(rewards_[i])

        if R_[i] == 1:
            choice_tm1 = 'R'
        elif G_[i] == 1:
            choice_tm1 = 'G'
        else:
            raise

    R = R_
    G = G_
    rateR = rateR_
    rateG = rateG_
    rewardR = rewardR_
    rewardG = rewardG_
    rewards = rewards_

    last_n = 825
    if len(R) > last_n:
        R = R[-last_n:]
        G = G[-last_n:]
        rateR = rateR[-last_n:]
        rateG = rateG[-last_n:]
        rewardR = rewardR[-last_n:]
        rewardG = rewardG[-last_n:]
        rewards = rewards[-last_n:]

    R = np.asarray(R)
    G = np.asarray(G)
    rateR = np.asarray(rateR)
    rateG = np.asarray(rateG)
    rewardR = np.asarray(rewardR)
    rewardG = np.asarray(rewardG)

    #x = np.arange(len(G))
    #y = np.arctan(np.cumsum(R)/np.cumsum(G)) * 180/np.pi
    #plot.plot(x, y, '-', color=Figure.colors('blue'), lw=lw, zorder=10)

    #scipy.ndimage.filters.gaussian_filter1d(input, sigma, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)[source]
    cumrewardR = np.cumsum(rewardR)
    cumrewardG = np.cumsum(rewardG)

    #cumrewardR = gaussian_filter1d(cumrewardR, sigma=6)
    #cumrewardG = gaussian_filter1d(cumrewardG, sigma=6)

    print(R_[:10])
    print(rewardR_[:10])
    print("")
    print(G_[:10])
    print(rewardG_[:10])
    print("")
    print(rewards[:10])

    rewardR = gaussian_filter1d(np.asarray(rewardR, dtype=float), sigma=6)
    rewardG = gaussian_filter1d(np.asarray(rewardG, dtype=float), sigma=6)

    R = gaussian_filter1d(np.asarray(R, dtype=float), sigma=6)
    G = gaussian_filter1d(np.asarray(G, dtype=float), sigma=6)

    x = np.arange(len(rewardR))
    y = np.arctan(rewardR/rewardG) * 180/np.pi
    plot.plot(x, y, '-', color=Figure.colors('black'), lw=lw, zorder=5)

    x = np.arange(len(R))
    y = np.arctan(R/G) * 180/np.pi
    plot.plot(x, y, '-', color=Figure.colors('blue'), lw=lw, zorder=8)

    # Exact -- replace with ratio
    start = 0
    for i in xrange(1, len(rateR)):
        if rateR[i] != rateR[i-1] or i == len(rateR)-1:
            xx = range(start, i)
            yy = rateR[xx]/rateG[xx]
            yy = np.arctan(yy) * 180/np.pi
            plot.plot(xx, yy, '-', color='k', lw=1.25*lw, zorder=4)

            start = i

    #xx = np.arange(len(rateR))
    #yy = np.arctan(rateR/rateG) * 180/np.pi
    #plot.plot(xx, yy, '-', color='k', lw=lw+0.25, zorder=4)

    plot.yticks([0, 45, 90])

    plot.xlim(0, last_n)

def sort_trials(trialsfile, figspath, name, **kwargs):
    """
    Sort trials.

    """
    # Load trials
    trials, U, Q, Z, A, R, M, perf, states = utils.load(trialsfile)

    # Data shape
    Ntime = states.shape[0]
    N     = states.shape[-1]

    # Same for every trial
    time = trials[0]['time']

    # Aligned time
    time_aligned = time - trials[0]['durations']['f1'][0]

    #=====================================================================================
    # Sort trials
    #=====================================================================================

    # Sort
    trials_by_cond = {}
    for n, trial in enumerate(trials):
        if not perf.decisions[n] and perf.corrects[n]:
            continue

        # Condition
        gt_lt = trial['gt_lt']
        fpair = trial['fpair']
        if gt_lt == '>':
            f1, f2 = fpair
        else:
            f2, f1 = fpair
        cond = (f1, f2)

        # States
        Mn = np.tile(M[:,n], (N,1)).T
        Sn = states[:,n]*Mn

        # Check firing rates
        assert np.all(Sn >= 0), Sn[np.where(Sn < 0)]
        assert np.all(Sn <= 1), Sn[np.where(Sn > 1)]

        trials_by_cond.setdefault(cond, {'r': np.zeros((Ntime, N)),
                                         'n': np.zeros((Ntime, N), dtype=int)})
        trials_by_cond[cond]['r'] += Sn
        trials_by_cond[cond]['n'] += Mn

    # Average
    for cond in trials_by_cond:
        trials_by_cond[cond] = utils.div(trials_by_cond[cond]['r'],
                                         trials_by_cond[cond]['n'])

    #=====================================================================================
    # Plot
    #=====================================================================================

    lw = kwargs.get('lw', 1.5)

    for unit in xrange(N):
        fig  = Figure()
        plot = fig.add()

        #---------------------------------------------------------------------------------

        for (f1, f2), r in trials_by_cond.items():
            plot.plot(time_aligned, r[:,unit], color=smap.to_rgba(f1), lw=lw)

        plot.xlim(time_aligned[0], time_aligned[-1])
        plot.ylim(0, 1)

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

    #=====================================================================================

    if action == 'cumulative-choice':
        fig  = Figure(axislabelsize=10, ticklabelsize=9)
        plot = fig.add()

        cumulative_choice(config['savefile'], plot)

        plot.xlabel('Cumulative green choices')
        plot.ylabel('Cumulative red choices')

        fig.save(path=config['figspath'], name='cumulative_choice')
        fig.close()

    if action == 'instantaneous-choice':
        fig  = Figure(axislabelsize=10, ticklabelsize=9)
        plot = fig.add()

        instantaneous_choice(config['savefile'], plot)

        plot.xlabel('Choice number')
        plot.ylabel('Slope')

        fig.save(path=config['figspath'], name='instantaneous_choice')
        fig.close()

    elif action in ['trials-b', 'trials-e']:
        try:
            trials_per_condition = int(args[0])
        except IndexError:
            trials_per_condition = 100

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'])
        m     = model.m

        #---------------------------------------------------------------------------------
        # Trial conditions
        #---------------------------------------------------------------------------------

        rng = np.random.RandomState(config['seed-trials'])

        n_conditions = m.n_conditions
        n_trials     = n_conditions * trials_per_condition
        print("Generating {} trial conditions ...".format(n_trials))

        # Task
        task = model.Task()

        trials = []
        for n in xrange(n_trials):
            trial = task.get_condition(rng, pg.config['dt'])
            trials.append(trial)

        #---------------------------------------------------------------------------------
        # Run trials
        #---------------------------------------------------------------------------------

        print("Running trials ...")
        if action == 'trials-b':
            print("Behavior only.")
            name = 'trials_behavior'
            (U, Q, Z, A, R, M, init, states_0,
             perf) = pg.run_trials(trials, progress_bar=True)
            save = [trials, A, R, M, perf]
        else:
            print("Behavior and neural activity.")
            name = 'trials_electrophysiology'
            (U, Q, Z, A, R, M, init, states_0, perf,
             states) = pg.run_trials(trials, return_states=True, progress_bar=True)
            save = [trials, U, Q, Z, A, R, M, perf, states]
        perf.display()

        #---------------------------------------------------------------------------------
        # Save trials
        #---------------------------------------------------------------------------------

        # Save
        print("Saving ...")
        trialsfile = os.path.join(config['scratchpath'], name + '.pkl')
        utils.save(trialsfile, save)

        # File size
        size_in_bytes = os.path.getsize(trialsfile)
        print("File size: {:.1f} MB".format(size_in_bytes/2**20))

    #=====================================================================================

    elif action == 'sort':
        trialsfile = os.path.join(config['scratchpath'], 'trials_electrophysiology.pkl')
        sort_trials(trialsfile, config['figspath'], 'sorted')
