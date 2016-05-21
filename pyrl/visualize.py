from __future__ import division

from pycog.figtools import Figure, mpl

def plot_trial(trial_info, trial, figspath, name):
    U, Z, A, R, M, init, states_0, perf = trial_info
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
    obs_time    = time[:t-1] + dt
    reward_time = act_time + dt
    xlim        = (0, max(time))

    #-------------------------------------------------------------------------------------
    # Observables
    #-------------------------------------------------------------------------------------

    plot = plots['observables']
    plot.plot(obs_time, U[:t-1,0], 'o', ms=5, mew=0, mfc=Figure.colors('blue'))
    plot.plot(obs_time, U[:t-1,0], lw=1.25, color=Figure.colors('blue'),   label='Keydown')
    plot.plot(obs_time, U[:t-1,1], 'o', ms=5, mew=0, mfc=Figure.colors('orange'))
    plot.plot(obs_time, U[:t-1,1], lw=1.25, color=Figure.colors('orange'), label=r'$f_\text{pos}$')
    plot.plot(obs_time, U[:t-1,2], 'o', ms=5, mew=0, mfc=Figure.colors('purple'))
    plot.plot(obs_time, U[:t-1,2], lw=1.25, color=Figure.colors('purple'), label=r'$f_\text{neg}$')
    plot.xlim(*xlim)
    plot.ylim(0, 1)
    plot.ylabel('Observables')

    if trial['gt_lt'] == '>':
        f1, f2 = trial['fpair']
    else:
        f2, f1 = trial['fpair']
    plot.text_upper_right(str((f1, f2)))

    #coh = trial['left_right']*trial['coh']
    #if coh < 0:
    #    color = Figure.colors('orange')
    #elif coh > 0:
    #    color = Figure.colors('purple')
    #else:
    #    color = Figure.colors('k')
    #plot.text_upper_right('Coh = {:.1f}\%'.format(coh), color=color)

    props = {'prop': {'size': 7}, 'handlelength': 1.2,
             'handletextpad': 1.2, 'labelspacing': 0.8}
    plot.legend(bbox_to_anchor=(1.2, 0.8), **props)

    #-------------------------------------------------------------------------------------
    # Policy
    #-------------------------------------------------------------------------------------

    plot = plots['policy']
    plot.plot(act_time, Z[:t,0], 'o', ms=5, mew=0, mfc=Figure.colors('blue'))
    plot.plot(act_time, Z[:t,0], lw=1.25, color=Figure.colors('blue'),
              label='Keydown')
    plot.plot(act_time, Z[:t,1], 'o', ms=5, mew=0, mfc=Figure.colors('orange'))
    plot.plot(act_time, Z[:t,1], lw=1.25, color=Figure.colors('orange'),
              label='$f_1 > f_2$')
    plot.plot(act_time, Z[:t,2], 'o', ms=5, mew=0, mfc=Figure.colors('purple'))
    plot.plot(act_time, Z[:t,2], lw=1.25, color=Figure.colors('purple'),
              label='$f_1 < f_2$')
    plot.xlim(*xlim)
    plot.ylim(0, 1)
    plot.ylabel('Action probabilities')

    props = {'prop': {'size': 7}, 'handlelength': 1.2,
             'handletextpad': 1.2, 'labelspacing': 0.8}
    plot.legend(bbox_to_anchor=(1.27, 0.8), **props)

    #-------------------------------------------------------------------------------------
    # Actions
    #-------------------------------------------------------------------------------------

    plot = plots['actions']
    actions = [np.argmax(a) for a in A[:t]]
    plot.plot(act_time, actions, 'o', ms=5, mew=0, mfc=Figure.colors('red'))
    plot.plot(act_time, actions, lw=1.25, color=Figure.colors('red'))
    plot.xlim(*xlim)
    plot.ylim(0, 2)
    plot.yticks([0, 1, 2])
    plot.yticklabels(['Keydown', '$f_1 > f_2$', '$f_1 < f_2$'])
    plot.ylabel('Action')

    #-------------------------------------------------------------------------------------
    # Rewards
    #-------------------------------------------------------------------------------------

    plot = plots['rewards']
    plot.plot(reward_time, R[:t], 'o', ms=5, mew=0, mfc=Figure.colors('red'))
    plot.plot(reward_time, R[:t], lw=1.25, color=Figure.colors('red'))
    plot.xlim(*xlim)
    plot.ylim(R_TERMINATE, R_CORRECT)
    plot.xlabel('Time (ms)')
    plot.ylabel('Reward')

    #-------------------------------------------------------------------------------------

    fig.save(path=figspath, name=name)
    fig.close()
