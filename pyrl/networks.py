from __future__ import absolute_import

from . import gru
from . import gru2
from . import simple

Networks = {
    'gru':    gru.GRU,
    'gru2':   gru2.GRU,
    'simple': simple.Simple
    }
