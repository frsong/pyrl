from __future__ import division

import os

import numpy as np

from scipy import stats

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

def indifference_point(trialsfile, offers, plot=None, **kwargs):
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

    X = []
    Y = []
    for i, offer in enumerate(offers):
        B, A = offer
        X.append((B - A)/(B + A))
        Y.append(pB_by_offer[offer])
    X = np.asarray(X)
    Y = np.asarray(Y)

    idx = np.argsort(X)
    X   = X[idx]
    Y   = Y[idx]

    #-------------------------------------------------------------------------------------
    # Fit
    #-------------------------------------------------------------------------------------

    try:
        popt, func = fittools.fit_psychometric(X, Y)

        mu   = popt['mu']
        idpt = (1+mu)/(1-mu)
        print("Indifference point = {}".format(idpt))

        fit_x = np.linspace(min(X), max(X), 201)
        fit_y = func(fit_x, **popt)
        fit   = fit_x, fit_y
    except RuntimeError:
        print("Unable to fit, drawing a line through the points.")
        mu   = None
        idpt = None
        fit  = X, Y

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    if plot is None:
        return idpt

    lw       = kwargs.get('lw', 1.5)
    ms       = kwargs.get('ms', 7)
    rotation = kwargs.get('rotation', 60)

    plot.plot(fit_x, 100*fit_y, '-', color='k', lw=lw)
    plot.plot(X, 100*Y, 'o', color='k', ms=ms)

    plot.hline(50, color='0.5', zorder=2)
    if mu is not None:
        plot.text_upper_left('1A = {:.1f}B'.format(idpt), fontsize=10)
        plot.vline(mu, color='0.5', zorder=2)

    #plot.xticks(range(len(offers)))
    #plot.xticklabels(['{}B:{}A'.format(*offer) for offer in offers], rotation=rotation)

    #plot.xlim(0, len(offers)-1)
    plot.ylim(0, 100)

    return idpt

#/////////////////////////////////////////////////////////////////////////////////////////

from scipy import stats

_figpath = '/Users/francis/Dropbox/Postdoc/code/git/frsong/pyrl/examples/temp'

CHOSEN_VALUE  = 0
OFFER_VALUE_A = 1
OFFER_VALUE_B = 2
CHOICE_A      = 3
CHOICE_B      = 4
CONSTANT      = 5
nreg          = 6

def classify_units(trials, perf, r, idpt):
    """
    Determine units' selectivity to offer value, choice value, and choice units.

    """
    def get_prechoice_firing_rate(trial, t_choice, r_trial):
        time     = trial['time']
        t_choice = time[t_choice]

        idx = np.where((-500+t_choice <= time) & (time < t_choice))

        return np.mean(r_trial[idx], axis=0)

    def rectify(x):
        return x*(x > 0)

    def step(x):
        return 1*(x > 0)

    valid_trials = [(n, trial) for n, trial in enumerate(trials)
                    if perf.choices[n] is not None
                    and trial['offer'][0] != 0 and trial['offer'][1] != 0]
    ntrials = len(valid_trials)
    print("Valid trials: {}".format(ntrials))

    nunits = r.shape[-1]
    #idx, = np.where(np.std(r, axis=(0, 1)) > 0.5)
    #active_units = np.arange(nunits)[idx]
    #print("Active units")
    #print(active_units)
    active_units = np.arange(nunits)
    nunits = len(active_units)

    x0 = (idpt - 1)/(idpt + 1)

    unit_types = {}
    for unit in active_units:
        Y = np.zeros(ntrials)
        X = {k: np.zeros(ntrials)
             for k in ['chosen-value', 'offer-value-A', 'offer-value-B', 'choice']}
        for i, (n, trial) in enumerate(valid_trials):
            B, A = trial['offer']
            x    = (B - A)/(B + A)

            X['chosen-value'][i]  = abs(x - x0)
            X['offer-value-A'][i] = -rectify(-(x - x0))
            X['offer-value-B'][i] = +rectify(+(x - x0))
            X['choice'][i]        = +1 if perf.choices[n] == 'B' else -1
            Y[i] = get_prechoice_firing_rate(trial, perf.t_choices[n], r[:,n,unit])

        psig  = 0.05
        corr2 = {}
        for k, v in X.items():
            corr, pval = stats.pearsonr(v, Y)
            if pval < psig and corr**2 >= 0.05:#0.1:
                corr2[k] = corr**2

                slope, intercept, r_value, p_value, std_err = stats.linregress(v, Y)
                assert np.isclose(corr**2, r_value**2)
                assert np.isclose(pval, p_value)

                '''
                fig  = Figure()
                plot = fig.add()

                x_fit = np.linspace(-1, 1, 201)
                y_fit = slope*x_fit + intercept
                plot.plot(x_fit, y_fit, color='k', lw=1, zorder=3)
                plot.plot(v, Y, 'o', ms=4, mfc='k', mec='w', mew=0.5)

                plot.text_upper_right(str(corr), zorder=5)
                plot.text_lower_right(str(pval), zorder=5)

                fig.save(path=_figpath, name='unit_type_{:03d}_{}'.format(unit, k))
                fig.close()
                '''

        # If there is a significant correlation, find the var with greatest correlation
        if corr2:
            unit_types[unit] = max(corr2, key=corr2.get)

    #for k in sorted(unit_types.keys()):
    #    print(k, unit_types[k])
    return unit_types

#/////////////////////////////////////////////////////////////////////////////////////////

def sort_epoch(behaviorfile, activityfile, epoch, offers, plots, units=None, network='p',
               separate_by_choice=False, **kwargs):
    """
    Sort trials.

    """
    # Load trials
    data = utils.load(activityfile)
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
    # Classify units
    #=====================================================================================

    idpt = indifference_point(behaviorfile, offers)
    unit_types = classify_units(trials, perf, r, idpt)
    #unit_types = {}

    numbers = {}
    for v in unit_types.values():
        numbers[v] = 0
    for k, v in unit_types.items():
        numbers[v] += 1

    n_tot = np.sum(numbers.values())
    for k, v in numbers.items():
        print("{}: {}/{} = {}%".format(k, v, n_tot, 100*v/n_tot))

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

            if unit in unit_types:
                plot.text_upper_right(unit_types[unit], fontsize=9)

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

    elif action == 'indifference_point':
        trialsfile = runtools.behaviorfile(config['trialspath'])

        fig  = Figure()
        plot = fig.add()

        spec = config['model'].spec

        indifference_point(trialsfile, spec.offers, plot)

        plot.xlabel('$(n_B - n_A)/(n_B + n_A)$')
        plot.ylabel('Percent choice B')

        #plot.text_upper_left('1A = {}B'.format(spec.A_to_B), fontsize=10)

        fig.save(path=config['figspath'], name=action)
        fig.close()

    #=====================================================================================

    elif action == 'sort_epoch':
        behaviorfile = runtools.behaviorfile(config['trialspath'])
        activityfile = runtools.activityfile(config['trialspath'])

        epoch = args[0]

        if 'value' in args:
            network = 'v'
        else:
            network = 'p'

        separate_by_choice = ('separate-by-choice' in args)

        sort_epoch(behaviorfile, activityfile, epoch, config['model'].spec.offers,
                   os.path.join(config['figspath'], 'sorted'),
                   network=network, separate_by_choice=separate_by_choice)
