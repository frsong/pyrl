from __future__ import absolute_import

from . import gru
#from . import gru2
from . import linear
from . import simple

Networks = {
    'linear': linear.Linear,
    'gru':    gru.GRU,
    #'gru2':   gru2.GRU,
    'simple': simple.Simple
    }
