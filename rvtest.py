import numpy as np

rng = np.random.RandomState(1)

N = 10000
X = rng.uniform(-0.1, 0.1, size=N)
Y = rng.normal(size=N)

from pyrl.figtools import Figure
fig = Figure()
plot = fig.add()

plot.hist(0*X+Y)

fig.save()
