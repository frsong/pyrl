from __future__ import division

import numpy as np

rng = np.random.RandomState(1)

N = 100000

k     = 4
theta = 1/k
X = rng.gamma(k, theta, size=N)

#k     = 2.5
#theta = 2/k
#Y = rng.gamma(k, theta, size=N)

from pyrl.figtools import Figure
fig = Figure()
plot = fig.add()

plot.hist(X, normed=False, alpha=0.5)
#plot.hist(Y, normed=False, color=Figure.colors('red'), alpha=0.5)

print(np.mean(X), 1)
print(np.var(X), 1/k)
print(np.std(X)/np.mean(X), 1/np.sqrt(k))

fig.save()
