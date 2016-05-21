from __future__ import absolute_import

from .gru    import GRU
from .simple import Simple

Networks = {
    'gru':    GRU,
    'simple': Simple
    }
