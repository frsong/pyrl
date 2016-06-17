from __future__ import absolute_import

from .gru    import GRU
from .gru2   import GRU2
from .simple import Simple

Networks = {
    'gru':    GRU,
    'gru2':   GRU2,
    'simple': Simple
    }
