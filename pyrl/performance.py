from __future__ import absolute_import, division

from collections import OrderedDict

from . import utils

class DefaultPerformance(object):
    def __init__(self):
        pass

    def update(self, trial, status):
        pass

    def display(output=True):
        pass

class Performance2AFC(object):
    def __init__(self):
        self.decisions = []
        self.corrects  = []
        self.choices   = []
        self.t_choices = []
        #self.rewards   = []

    def update(self, trial, status):
        #self.rewards.append(reward)
        if 'correct' in status:
            self.decisions.append(True)
            self.corrects.append(status['correct'])
            if 'choice' in status:
                self.choices.append(status['choice'])
            else:
                self.choices.append(None)
            if 't_choice' in status:
                self.t_choices.append(status['t_choice'])
            else:
                self.t_choices.append(None)
        else:
            self.decisions.append(False)
            self.corrects.append(False)
            self.choices.append(None)
            self.t_choices.append(None)

    @property
    def n_trials(self):
        return len(self.decisions)

    @property
    def n_decision(self):
        return sum(self.decisions)

    @property
    def n_correct(self):
        return sum(self.corrects)

    def display(self, output=True):
        n_trials   = self.n_trials
        n_decision = self.n_decision
        n_correct  = self.n_correct

        items = OrderedDict()
        items['P(choice)'] = '{}/{} = {:.3f}'.format(n_decision, n_trials,
                                                     n_decision/n_trials)
        if n_decision > 0:
            items['P(correct|choice)'] = '{}/{} = {:.3f}'.format(n_correct, n_decision,
                                                                 n_correct/n_decision)

        if output:
            utils.print_dict(items)
        return items

class PerformancePostdecisionWager(object):
    def __init__(self):
        self.wagers    = []
        self.corrects  = []
        self.choices   = []
        self.t_choices = []

    def update(self, trial, status):
        self.wagers.append(trial['wager'])
        self.corrects.append(status.get('correct'))
        self.choices.append(status.get('choice'))
        self.t_choices.append(status.get('t_choice'))

    @property
    def n_correct(self):
        return sum([c for c in self.corrects if c is not None])

    @property
    def n_sure_decision(self):
        return len([1 for w, c in zip(self.wagers, self.choices) if w and c is not None])

    @property
    def n_trials(self):
        return len(self.choices)

    @property
    def n_decision(self):
        return len([1 for c in self.choices if c in ['L', 'R']])

    @property
    def n_sure(self):
        return len([1 for c in self.choices if c == 'S'])

    @property
    def n_answer(self):
        return len([1 for c in self.choices if c is not None])

    @property
    def n_wager(self):
        return sum(self.wagers)

    def display(self, output=True):
        n_trials        = self.n_trials
        n_decision      = self.n_decision
        n_correct       = self.n_correct
        n_sure_decision = self.n_sure_decision
        n_sure          = self.n_sure
        n_answer        = self.n_answer
        n_wager         = self.n_wager

        items = OrderedDict()
        items['P(answer)'] = '{}/{} = {:.3f}'.format(n_answer, n_trials,
                                                     n_answer/n_trials)
        items['P(decision)'] = '{}/{} = {:.3f}'.format(n_decision, n_trials,
                                                       n_decision/n_trials)
        if n_decision > 0:
            items['P(correct|decision)'] = '{}/{} = {:.3f}'.format(n_correct, n_decision,
                                                                   n_correct/n_decision)
        items['P(wager trials)'] = '{}/{} = {:.3f}'.format(n_wager, n_trials,
                                                           n_wager/n_trials)
        if n_sure_decision > 0:
            items['P(sure)'] = '{}/{} = {:.3f}'.format(n_sure, n_sure_decision,
                                                       n_sure/n_sure_decision)

        if output:
            utils.print_dict(items)
        return items
