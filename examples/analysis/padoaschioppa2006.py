from __future__ import division

import os

import numpy as np

from pyrl          import datatools, fittools, runtools, tasktools, utils
from pyrl.figtools import Figure

#/////////////////////////////////////////////////////////////////////////////////////////

def choice_pattern(trialsfile, offers, plot, **kwargs):
    # Load trials
    trials, A, R, M, perf = utils.load(trialsfile)

    B_by_offer    = {}
    n_nondecision = 0
    for n, trial in enumerate(trials):
        if perf.choices[n] is None:
            n_nondecision += 1
            continue

        juice_L, juice_R = trial['juice']
        offer = trial['offer']

        if perf.choices[n] == 'B':
            B = 1
        elif perf.choices[n] == 'A':
            B = 0
        else:
            raise ValueError("invalid choice")

        B_by_offer.setdefault(offer, []).append(B)
    print("Non-decision trials: {}/{}".format(n_nondecision, len(trials)))

    pB_by_offer = {}
    for offer in B_by_offer:
        Bs = B_by_offer[offer]
        pB_by_offer[offer] = utils.divide(sum(Bs), len(Bs))
        #print(offer, pB_by_offer[offer])

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    if plot is None:
        return

    ms       = kwargs.get('ms', 7)
    rotation = kwargs.get('rotation', 60)

    for i, offer in enumerate(offers):
        plot.plot(i, 100*pB_by_offer[offer], 'o', color='k', ms=ms)

    plot.xticks(range(len(offers)))
    plot.xticklabels(['{}B:{}A'.format(*offer) for offer in offers], rotation=rotation)

    plot.xlim(0, len(offers)-1)
    plot.ylim(0, 100)

#/////////////////////////////////////////////////////////////////////////////////////////

def classify_units(epochs_by_cond):
    """
    Determine units' selectivity to offer value, choice value, and choice units.

    """
    unit_types = []

    return unit_types

#/////////////////////////////////////////////////////////////////////////////////////////

def sort_epoch(trialsfile, epoch, offers, plots, units=None, network='p',
               separate_by_choice=False, **kwargs):
    """
    Sort trials.

    """
    # Load trials
    data = utils.load(trialsfile)
    if len(data) == 9:
        trials, U, Z, A, P, M, perf, r_p, r_v = data
    else:
        trials, U, Z, Z_b, A, P, M, perf, r_p, r_v = data

    if network == 'p':
        print("POLICY NETWORK")
        r = r_p
    else:
        print("VALUE NETWORK")
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
    # Sort trials
    #=====================================================================================

    # Epochs
    events = ['offer', 'choice']

    # Sort
    events_by_cond = {e: {} for e in events}
    n_by_cond      = {}
    n_nondecision  = 0
    for n, trial in enumerate(trials):
        if perf.choices[n] is None:
            n_nondecision += 1
            continue

        # Condition
        offer  = trial['offer']
        choice = perf.choices[n]

        if separate_by_choice:
            cond = (offer, choice)
        else:
            cond = offer

        n_by_cond.setdefault(cond, 0)
        n_by_cond[cond] += 1

        # Storage
        for e in events_by_cond:
            events_by_cond[e].setdefault(cond, {'r': np.zeros((Ntime_a, N)),
                                                'n': np.zeros((Ntime_a, N))})

        # Firing rates
        m_n = np.tile(M[:,n], (N,1)).T
        r_n = r[:,n]*m_n

        for e in events_by_cond:
            # Align point
            if e == 'offer':
                t0 = trial['epochs']['offer-on'][0]
            elif e == 'choice':
                t0 = perf.t_choices[n]
            else:
                raise ValueError(e)

            # Before
            n_b = r_n[:t0].shape[0]
            events_by_cond[e][cond]['r'][Ntime-1-n_b:Ntime-1] += r_n[:t0]
            events_by_cond[e][cond]['n'][Ntime-1-n_b:Ntime-1] += m_n[:t0]

            # After
            n_a = r_n[t0:].shape[0]
            events_by_cond[e][cond]['r'][Ntime-1:Ntime-1+n_a] += r_n[t0:]
            events_by_cond[e][cond]['n'][Ntime-1:Ntime-1+n_a] += m_n[t0:]
    print("Non-decision trials: {}/{}".format(n_nondecision, len(trials)))

    # Average trials
    for e in events_by_cond:
        for cond in events_by_cond[e]:
            events_by_cond[e][cond] = utils.div(events_by_cond[e][cond]['r'],
                                                events_by_cond[e][cond]['n'])

    # Epochs
    epochs = ['preoffer', 'postoffer', 'latedelay', 'prechoice']

    # Average epochs
    epochs_by_cond = {e: {} for e in epochs}
    for e in epochs_by_cond:
        if e == 'preoffer':
            ev = 'offer'
            w, = np.where((-500 <= time_a) & (time_a < 0))
        elif e == 'postoffer':
            ev = 'offer'
            w, = np.where((0 <= time_a) & (time_a < 500))
        elif e == 'latedelay':
            ev = 'offer'
            w, = np.where((500 <= time_a) & (time_a < 1000))
        elif e == 'prechoice':
            ev = 'choice'
            w, = np.where((-500 <= time_a) & (time_a < 0))
        else:
            raise ValueError(e)

        for cond in events_by_cond[ev]:
            epochs_by_cond[e][cond] = np.mean(events_by_cond[ev][cond][w], axis=0)

    #=====================================================================================
    # Plot
    #=====================================================================================

    lw  = kwargs.get('lw',  1.5)
    ms  = kwargs.get('ms',  6)
    mew = kwargs.get('mew', 0.5)
    rotation = kwargs.get('rotation', 60)
    #min_trials = kwargs.get('min_trials', 100)

    def plot_activity(plot, unit):
        yall = [1]

        min_trials = 20

        # Pre-offer
        epoch_by_cond = epochs_by_cond['preoffer']
        color = '0.7'
        if separate_by_choice:
            for choice, marker in zip(['A', 'B'], ['d', 'o']):
                x = []
                y = []
                for i, offer in enumerate(offers):
                    cond = (offer, choice)
                    if cond in n_by_cond and n_by_cond[cond] >= min_trials:
                        y_i = epoch_by_cond[cond][unit]
                        plot.plot(i, y_i, marker, mfc=color, mec=color, ms=0.8*ms,
                                  mew=0.8*mew, zorder=10)
                        yall.append(y_i)
                        if i != 0 and i != len(offers)-1:
                            x.append(i)
                            y.append(y_i)
                plot.plot(x, y, '-', color=color, lw=0.8*lw, zorder=5)
        else:
            x = []
            y = []
            for i, offer in enumerate(offers):
                y_i = epoch_by_cond[offer][unit]
                plot.plot(i, y_i, 'o', mfc=color, mec=color, ms=0.8*ms,
                          mew=0.8*mew, zorder=10)
                yall.append(y_i)
                if i != 0 and i != len(offers)-1:
                    x.append(i)
                    y.append(y_i)
            plot.plot(x, y, '-', color=color, lw=0.8*lw, zorder=5)

        # Epoch
        epoch_by_cond = epochs_by_cond[epoch]
        if epoch == 'postoffer':
            color = Figure.colors('darkblue')
        elif epoch == 'latedelay':
            color = Figure.colors('darkblue')
        elif epoch == 'prechoice':
            color = Figure.colors('darkblue')
        else:
            raise ValueError(epoch)
        if separate_by_choice:
            for choice, marker, color in zip(['A', 'B'], ['d', 'o'], [Figure.colors('red'), Figure.colors('blue')]):
                x = []
                y = []
                for i, offer in enumerate(offers):
                    cond = (offer, choice)
                    if cond in n_by_cond and n_by_cond[cond] >= min_trials:
                        y_i = epoch_by_cond[cond][unit]
                        yall.append(y_i)
                        plot.plot(i, y_i, marker, mfc=color, mec=color, ms=ms, mew=mew, zorder=10)
                        if i != 0 and i != len(offers)-1:
                            x.append(i)
                            y.append(y_i)
                plot.plot(x, y, '-', color=color, lw=lw, zorder=5)
        else:
            x = []
            y = []
            for i, offer in enumerate(offers):
                y_i = epoch_by_cond[offer][unit]
                plot.plot(i, y_i, 'o', mfc=color, mec=color, ms=ms, mew=mew, zorder=10)
                yall.append(y_i)
                if i != 0 and i != len(offers)-1:
                    x.append(i)
                    y.append(y_i)
            plot.plot(x, y, '-', color=color, lw=lw, zorder=5)

        plot.xticks(range(len(offers)))
        plot.xticklabels(['{}B:{}A'.format(*offer) for offer in offers],
                         rotation=rotation)

        plot.xlim(0, len(offers)-1)
        plot.lim('y', yall, lower=0)

        return yall

    #-------------------------------------------------------------------------------------

    if units is not None:
        for plot, unit in zip(plots, units):
            plot_activity(plot, unit)
    else:
        name = plots
        for unit in xrange(N):
            fig  = Figure()
            plot = fig.add()

            plot_activity(plot, unit)

            if separate_by_choice:
                suffix = '_sbc'
            else:
                suffix = ''

            fig.save(name+'_{}{}_{}{:03d}'.format(epoch, suffix, network, unit))
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
        except IndexError:
            trials_per_condition = 100

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'], config['dt'])

        spec         = model.spec
        juices       = spec.juices
        offers       = spec.offers
        n_conditions = spec.n_conditions
        n_trials     = trials_per_condition * n_conditions

        print("{} trials".format(n_trials))
        task   = model.Task()
        trials = []
        for n in xrange(n_trials):
            k = tasktools.unravel_index(n, (len(juices), len(offers)))
            context = {
                'juice': juices[k.pop(0)],
                'offer': offers[k.pop(0)]
                }
            trials.append(task.get_condition(pg.rng, pg.dt, context))
        runtools.run(action, trials, pg, config['trialspath'])

    #=====================================================================================

    elif action == 'choice_pattern':
        trialsfile = runtools.behaviorfile(config['trialspath'])

        fig  = Figure()
        plot = fig.add()

        spec = config['model'].spec

        choice_pattern(trialsfile, spec.offers, plot)

        plot.xlabel('Offer (\#B : \#A)')
        plot.ylabel('Percent choice B')

        plot.text_upper_left('1A = {}B'.format(spec.A_to_B), fontsize=10)

        fig.save(path=config['figspath'], name=action)
        fig.close()

    #=====================================================================================

    elif action == 'sort_epoch':
        trialsfile = runtools.activityfile(config['trialspath'])

        epoch = args[0]

        if 'value' in args:
            network = 'v'
        else:
            network = 'p'

        separate_by_choice = ('separate-by-choice' in args)

        sort_epoch(trialsfile, epoch, config['model'].spec.offers,
                   os.path.join(config['figspath'], 'sorted'),
                   network=network, separate_by_choice=separate_by_choice)
