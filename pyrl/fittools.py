from __future__ import division

import inspect
from collections import OrderedDict

import numpy       as np
import scipy.stats as stats

from scipy.optimize import curve_fit, fmin_l_bfgs_b as fminimize
from scipy.special  import erf

#=========================================================================================
# Binomial regression
#=========================================================================================

def binregress_objective(theta, x, y, func):
    p = func(x, *tuple(theta))
    w = np.where((p > 0) & (p < 1))[0]
    assert len(w) > 0, theta

    psafe = p[w]
    ysafe = y[w]

    return -sum(ysafe*np.log(psafe) + (1 - ysafe)*np.log(1 - psafe))/len(ysafe)

def binregress(x, y, func, theta_init, bounds=None):
    xmin, fmin, info = fminimize(binregress_objective, theta_init, bounds=bounds,
                                 args=(x, y, func), approx_grad=True, iprint=0)

    return xmin

#=========================================================================================
# Fit functions
#=========================================================================================

def weibull(x, alpha=1, beta=1):
    return 1 - 0.5*np.exp(-(x/alpha)**beta)

def cdf_gaussian(x, mu=0, sigma=1):
    return stats.norm.cdf(x, mu, sigma)

def cdf_gaussian_with_guessing(x, mu=0, sigma=1, gamma=0.1):
    return gamma + (1 - 2*gamma)*stats.norm.cdf(x, mu, sigma)

fit_functions = {
    'cdf_gaussian':             cdf_gaussian,
    'cdf_gaussin_with_gussing': cdf_gaussian_with_guessing,
    'weibull':                  weibull
    }

#=========================================================================================

def fit_psychometric(xdata, ydata, func=None, p0=None):
    """
    Fit a psychometric function.

    """
    if func is None:
        func = 'cdf_gaussian'

    if p0 is None:
        if func == 'cdf_gaussian':
            p0 = [np.mean(xdata), np.std(xdata)]
        elif func == 'cdf_gaussian_with_guessing':
            p0 = [np.mean(xdata), np.std(xdata), 0.1]
        else:
            raise ValueError("[ pycog.fittools.fit_psychometric ] Need initial guess p0.")
    if isinstance(func, str):
        func = fit_functions[func]

    popt_list, pcov_list = curve_fit(func, xdata, ydata, p0=p0)

    # Return parameters with names
    args = inspect.getargspec(func).args
    popt = OrderedDict()
    for name, value in zip(args[1:], popt_list):
        popt[name] = value

    return popt, func
