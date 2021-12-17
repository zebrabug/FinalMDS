"""
Module provide several methods for tail index estimaition
"""

import math
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def hill_estimator(x, k_range=None) -> np.ndarray:
    '''
    Tail index estimation using Hill method
    Function calculate Hill estimations
        H(k, n) = 1/k SUM(log(X[n-i+1]) - log(X[n-k]) for i in [1,k])
        H -> 1/Tail index when n -> inf
        Hill's tail index estimation
        Tail index = 1 / AVG(H) for k in [k1, k2]
        where k1 and k2 chosen empirically (usually based on graph of H(k)
    Function returns np.array of 1/H(k)
    '''
    
    x = np.sort(x)
    n = len(x)

    if k_range is None:
        k_range = range(2, int(n/2))
        
    H = [k / (np.sum(np.log(x[n-k+1:])) - k*(math.log(x[n-k]))) for k in k_range]
    return np.array(H)


def WLS(x, x_min):
    if x_min is None:
        x_min = x.min()
    else:
        x = x[x > x_min]
    return -np.sum(np.log(x.rank()/x.shape[0])) / np.sum(np.log(x/x_min))
 
    
def MLE(x, x_min=None):
    '''
    Maximum Likelihood Estimation for tail index
    Tail index = )
    Unbiased estimation needs multiplication on (N-2)/N
    '''
    if x_min is None:
        x_min = np.min(x)  # MLE estimation for x_min
    x = x[x > x_min]
    return x.shape[0] / np.sum(np.log(x/x_min))


def MoM(x, x_min=None):
    '''
    Momemtum estimation of tail index
    MoM = SUM(X) / (SUM(X) - N*x_min)
    Works only if tail index > 1
    '''
    if x_min is None:
        x_min = np.min(x)
    x = x[x > x_min]
    return np.mean(x) / (np.mean(x) - x_min)


def PM(x, x_min=None, method='PM'):
    '''
    Percentile methods for tail index estimation
    E.g. PM: tail = log(3) / log(P75) - log(P25)
    where P75 and P25 - 75 and 25 percentile correspondingly
    '''
    if x_min:
        x = x[x > x_min]
    
    if method == 'PM':
        p_high, p_low = np.quantile(x, [0.75, 0.25])
        return math.log(3) / (math.log(p_high) - math.log(p_low))
    elif method == 'MPM':
        p_high, p_low = np.quantile(x, [0.75, 0.5])
        return math.log(2) / (math.log(p_high) - math.log(p_low))
    elif method == 'GMPM':
        p_high = np.quantile(x, 0.75)
        return (1 - math.log(4)) / (np.mean(np.log(x)) - math.log(p_high))
    else:
        raise ValueError('Incorrect method. Should be PM or MPM or GMPM')


def CCDF_tail_regressor(x, data=None, bins=None, binwidth=1, x_range=None, verbose=True):
    '''
    Function calculate CCDF and linear regression in log-log scale
    data - DataFrame or Series and x is column/series name
    OR x is numpy ndarray
    '''
    if data is not None:
        x = data[x]
        
    if bins is None:
        bins = int((x.max() - x.min()) / binwidth)
        
    # pdf estimation
    npdf, xs = np.histogram(x, bins=bins)
    
    # ccdf
    ccdf = 1. - np.cumsum(npdf)/np.sum(npdf)
    
    dxy = pd.DataFrame({'X': xs[1:], 'Y': ccdf})  # for statsmodels, temp DataFrame
    if x_range is not None:
        dxy = dxy[(dxy.X >= x_range[0]) & (dxy.X <= x_range[1])]
    
    model = smf.gls('np.log(Y) ~ np.log(X)', data=dxy)  # linear regression
    res = model.fit()

    if verbose:
        return res.params.iloc[1], xs[1:], ccdf, res  # return 
    else:
        return res.params.iloc[1]
        
        
def main():
    pass

if __name__ == '__main__':
    main()