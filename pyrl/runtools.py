from __future__ import absolute_import

import os

from . import utils

def behaviorfile(path):
    return os.path.join(path, 'trials_behavior.pkl')

def activityfile(path):
    return os.path.join(path, 'trials_activity.pkl')

def run(action, trials, pg, scratchpath, dt_save=None):
    if dt_save is not None:
        dt  = pg.dt
        inc = int(dt_save/dt)
    else:
        inc = 1
    print("Saving in increments of {}".format(inc))

    # Run trials
    if action == 'trials-b':
        print("Saving behavior only.")
        trialsfile = behaviorfile(scratchpath)

        (U, Q, Q_b, Z, Z_b, A, R, M, init, init_b, states_0, states_0_b,
         perf) = pg.run_trials(trials, progress_bar=True)

        for trial in trials:
            trial['time'] = trial['time'][::inc]
        save = [trials, A[::inc], R[::inc], M[::inc], perf]
    elif action == 'trials-a':
        print("Saving behavior + activity.")
        trialsfile = activityfile(scratchpath)

        (U, Q, Q_b, Z, Z_b, A, R, M, init, init_b, states_0, states_0_b,
         perf, states, states_b) = pg.run_trials(trials,
                                                 return_states=True,
                                                 progress_bar=True)

        for trial in trials:
            trial['time'] = trial['time'][::inc]
        save = [trials, U[::inc], Z[::inc], Z_b[::inc], A[::inc], R[::inc],
                M[::inc], perf, states[::inc], states_b[::inc]]
    else:
        raise ValueError(action)

    # Performance
    perf.display()

    # Save
    utils.save(trialsfile, save)

    # File size
    size_in_bytes = os.path.getsize(trialsfile)
    print("File size: {:.1f} MB".format(size_in_bytes/2**20))
