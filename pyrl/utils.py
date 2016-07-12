from __future__ import division

import cPickle as pickle
import datetime
import errno
import os
import signal
import subprocess
import sys

from collections import OrderedDict

import numpy as np

def println(line):
    sys.stdout.write(line)
    sys.stdout.flush()

def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def savelist(filename, list_):
    with open(filename, 'w') as f:
        for item in list_:
            f.write('{}\n'.format(item))

def loadlist(filename):
    with open(filename) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def print_dict(settings_, indent=2, title=None):
    """
    Pretty print.

    """
    if isinstance(settings_, (list, tuple)):
        settings = OrderedDict()
        for s in settings_:
            settings.update(s)
    else:
        settings = settings_

    if title is not None:
        print(title)
    maxlen = max([len(s) for s in settings])
    for k, v in settings.items():
        print(indent*' ' + '{}:{}{}'.format(k, (maxlen - len(k) + 1)*' ', v))
    sys.stdout.flush()

def copy_to_clipboard(s):
    try:
        proc = subprocess.Popen('pbcopy', env={'LANG': 'en_US.UTF-8'},
                                stdin=subprocess.PIPE)
        proc.communicate(s.encode('utf-8'))
    except:
        pass

#=========================================================================================
# Safe division
#=========================================================================================

def div(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.true_divide(x, y)
        z[z == np.inf] = 0
        z = np.nan_to_num(z)

    return z

def divide(x, y):
    try:
        z = x/y
        if np.isnan(z):
            raise ZeroDivisionError
        return z
    except ZeroDivisionError:
        return 0

#=========================================================================================
# Paths
#=========================================================================================

def get_here(file):
    return os.path.abspath(os.path.dirname(file))

def get_parent(dir):
    return os.path.abspath(os.path.join(dir, os.pardir))

#=========================================================================================
# Files
#=========================================================================================



#=========================================================================================
# Pickle
#=========================================================================================

def save(filename, obj):
    """
    Disable keyboard interrupt while pickling.

    """
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    signal.signal(signal.SIGINT, s)

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

#=========================================================================================
# Unit conversion
#=========================================================================================

def mm_to_inch(mm):
    return mm/25.4

#=========================================================================================
# Reporting
#=========================================================================================

def elapsed_time(tstart):
    tnow      = datetime.datetime.now()
    totalsecs = (tnow - tstart).total_seconds()

    hrs  = int(totalsecs//3600)
    mins = int(totalsecs%3600)//60
    secs = int(totalsecs%60)

    return '{}h {}m {}s elapsed'.format(hrs, mins, secs)
