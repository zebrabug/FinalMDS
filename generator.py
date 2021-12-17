import datetime as dt
import pickle

import pandas as pd
import numpy as np
import numba


# ===============================================================================================================
from sklearn.neighbors import KernelDensity


# Wrapper class for Kernel Density Estimator
class KDEsampler:
    def __init__(self, ticksize):
        self.ticksize = ticksize
        self.kde = KernelDensity(bandwidth=1., kernel='tophat')

    def fit(self, X):
        Y = self.preprocess(X)
        self.kde.fit(Y)
        return self

    def sample(self, n):
        Y = self.kde.sample(n)
        return self.postprocess(Y)

    def save(self, filename):
        """
        Dump model using pickle to file
        :param filename: file where to dump
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


    def preprocess(self, X):
        """
        Preprocess time-series for KDE
        :param X: 2D sample, [volume, impact]
        :return: new time-series Y
        """
        Y = np.zeros_like(X)
        Y[:, 0] = np.log(X[:, 0])
        xmin, xmax = np.min(Y[:, 0]), np.max(Y[:, 0])  # range of volumes

        # Impact -> Ticks -> x2+1 (to differentiate tick from each other)
        Y[:, 1] = (np.round(X[:, 1] / self.ticksize)) * 2 + 1
        ymin, ymax = np.min(Y[:, 1]), np.max(Y[:, 1])  # range of impacts
        self.scale = round((ymax - ymin) / (xmax - xmin))  # scale to make Volume and Impact comparable
        Y[:, 0] = Y[:, 0] * self.scale
        return Y

    def postprocess(self, Y):
        """
        Inverse transform to preprocess
        :param Y: input 2D time-series
        :return: X time-series [volume, impact]
        """
        X = np.zeros_like(Y)
        X[:, 0] = np.round(np.exp(Y[:, 0] / self.scale))
        X[:, 1] = np.floor(Y[:, 1] / 2) * self.ticksize
        return X


# ==========================================================================================================
# some auxiliary functions

@numba.njit(fastmath=True)
def _generate_exp_series(n, intensity,  max_value):
    """
    Generate series from exponential distribution
    with defined intensity (time intervals between events of Poisson process)
    Overall time (sum of time-series) < max_value
    Returns cumulative sum (series of time) and intervals itself
    """
    x = np.random.exponential(1, size=n) / intensity
    return np.cumsum(x[np.cumsum(x) < max_value]), \
           x[np.cumsum(x) < max_value]


@numba.njit(fastmath=True)
def _generate_price_series(time_deltas, mu, sigma, start_price, ticksize=0.):
    """
    Simple Brownian motion with drift dS = mu*dt + sigma*dZ
    time_deltas: array of dt
    :param mu, sigma: parameters of process
    :param ticksize: used for price rounding (if set)
    """
    prices = start_price + np.cumsum(mu*time_deltas +
                                     sigma * time_deltas**0.5 * np.random.randn(time_deltas.shape[0]))
    if ticksize == 0.:
        return prices
    else:
        return np.round_(prices/ticksize, 0, prices) * ticksize


def _preprocess_choice_alias(probs):
    """
    Alias method for fast weighted random generation (see Wikipedia)
    Based on https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087
    Preprocess probabilities (weights)
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)

    smaller, larger = [], []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small, large = smaller.pop(), larger.pop()
        J[small] = large
        q[large] = q[large] - (1.0 - q[small])
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return q, J

@numba.njit(fastmath=True)
def _choice_alias_method(n, q, J):
    """
    Analog of numpy choice with numba using alias method
    Based on https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087
    """
    r_table, r_alias = np.random.rand(n), np.random.rand(n)
    res = np.zeros(n, dtype=np.int32)
    lj = len(J)
    for i in range(n):
        kk = int(np.floor(r_table[i]*lj))
        if r_alias[i] < q[kk]:
            res[i] = kk
        else:
            res[i] = J[kk]
    return res

# Spread sampler using alias method and numba
# Preferable for very large sample size
@numba.njit(fastmath=True)
def _generate_spread_series_alias(n, ticksize, q, J):
    return (_choice_alias_method(n, q, J) + 1) * ticksize


def _generate_spread_series(n, ticksize, probs):
    """
    Spread sampler from spread PMF
    :param n: sample size
    :param ticksize:
    :param probs: array (list or ndarray) of probabilities of spreads
    probs[i] = probability(spread == ticksize * (i+1)]
    :return: spread sample
    """
    return (np.random.choice(len(probs), n, p=probs) + 1) * ticksize


def _generate_Vx0_tuples(n, prob_matrix, use_numba=False):
    """
    Generation two-dimensional discrete random variable
    from given PMF matrix (prob_matrix)
    :param n: sample size
    :param prob_matrix: PMF of distribuntion
    :param use_numba: use alias method and numba (not recommended)
    :return: sample 2D series (volume, impact)
    """
    probs = prob_matrix.flatten()
    m = prob_matrix.shape[1]
    if use_numba:
        q, J = _preprocess_choice_alias(probs)
        z = _choice_alias_method(n, q, J)
    else:
        z = np.random.choice(len(probs), size=n, p=probs)
    return np.array([(x % m, int(x // m)) for x in z])


# ===========================================================================================================
# main class
class Market:
    def __init__(self, mu, sigma, ticksize, spread_pmf, basis='hour'):
        """
        Initialize market parameters
        :param mu, sigma: parameters of brownian motion (drift and volatility) for basis
        :param spread_pmf: PMF of spreads, starts from 1 tick, list or numpy.ndarray
        :param basis: hour (default) or minute
        """
        self.mu = mu  # market drift (dayly basis)
        self.sigma = sigma  # market std dev (daily basis)

        self.ticksize = ticksize
        self.basis = basis
        if self.basis == 'hour':
            self.base = 3600
        elif self.basis == 'minute':
            self.base = 60
        else:
            raise ValueError("Unknown basis. Use hour or minute.")

        self.spread_pmf = np.array(spread_pmf) / np.sum(spread_pmf)  # normalization
        self._q, self._J = _preprocess_choice_alias(self.spread_pmf)  # alias preprocess


    def init_x0_model(self, filename):
        """
        Load PMF for x0 volumes/impact distribution from numpy array saved to npz file
        :return: self
        """
        arrays = np.load(filename)
        self.x0_pmf = arrays['arr_0']  # first array is pmf
        self.x0_volumes = arrays['arr_1']  # second array is values for volume indexes
        return self

    def init_xx_model(self, filename):
        """
        Load KDE model for volumes/impact continuous distribution
        from pickle file with KDE_sampler object
        :return: self
        """
        with open(filename, 'rb') as f:
            self.kde_model = pickle.load(f)  # KDEsampler instance
        return self


    def generate_order_times(self, intensity, start_time, end_time) -> (np.ndarray, np.ndarray):
        """
        Generate Poisson process event time
        :param intensity: orders per basis period
        :param start_time, end_time: period of generation, datetime or time (for intraday)
        :return: series of event times (t), series of time intervals (dt)
        """
        if isinstance(start_time, dt.time):  # time -> datetime
            start_time = dt.datetime.combine(dt.datetime.today(), start_time)
            end_time = dt.datetime.combine(dt.datetime.today(), end_time)

        time_delta = (end_time - start_time).seconds / self.base
        deltas_cum, deltas = _generate_exp_series(int(2.5 * intensity * time_delta), intensity, time_delta)

        if self.basis == 'hour':
            return np.array([start_time + dt.timedelta(hours=d) for d in deltas_cum]), deltas
        elif self.basis == 'minute':
            return np.array([start_time + dt.timedelta(minutes=d) for d in deltas_cum]), deltas
        else:
            raise ValueError('Unknown basis. Use hour or minute.')


    def generate_bid_ask_prices(self, time_deltas, start_bid_price):
        """
        Generate bid/ask prices time-series using Brownian motion process (for bid)
        and spread generated series for ask
        :param time_deltas: dt array
        :return: bid and ask series
        """
        bids = _generate_price_series(time_deltas, self.mu, self.sigma, start_bid_price, self.ticksize)
        asks = bids + _generate_spread_series(time_deltas.shape[0], self.ticksize, self.spread_pmf)
        return bids, asks


    def generate_market_orders_x0(self, n):
        """
        Generate sample (volume, impact) for discrete part of volumes distribution
        :param n: sample size
        :return: 2D sample (volumes, impacts)
        """
        x0_tuples = _generate_Vx0_tuples(n, self.x0_pmf)
        res = np.zeros_like(x0_tuples, dtype=np.float64)
        res[:, 0] = self.x0_volumes[x0_tuples[:, 0]]  # transform indexes to volumes
        res[:, 1] = x0_tuples[:, 1] * self.ticksize  # transfrom ticks to impact values
        return res


    def generate_market_orders_xx(self, n):
        """
        Generate 2D sample (volume, impact) from continuous part of volumes distribution
        :param n: sample size
        :return: 2D sample (volumes, impacts)
        """
        return self.kde_model.sample(n)


    def generate_market_orders(self, n, buy_share=0.5, Vx0_share=0.):
        """
        :param n: sample size
        :param buy_share: share of buy orders, default is 50%
        :param Vx0_share: share of x00 market orders, default is 0%
        :return: side, volume, impact arrays
        """
        side = 2 * (np.random.rand(n) > buy_share) - 1  # 1 = BUY, -1 = SELL
        Vx0_mask = (np.random.rand(n) > Vx0_share)  # True is x00 volume order

        z0 = self.generate_market_orders_x0(n)
        zx = self.generate_market_orders_xx(n)

        Vx0_mask = np.vstack((Vx0_mask, Vx0_mask)).T
        z = np.where(Vx0_mask, z0, zx)
        return side, z[:, 0], z[:, 1]


    def order_book(self, intensity, start_price, start_time, end_time, buy_share=0.5, Vx0_share=0.):
        """
        Generate market order book
        :param intensity: Poisson process intensity
        :param start_price: start price for Brownian motion process
        :param start_time, end_time: start and end points of generation
        :param buy_share: share of buy market orders
        :param Vx0_share: share of x00 volumes in volumes distribution
        :return: dataframe ['Time', 'Side', 'Volume', 'Impact', 'Bid', 'Ask']
        """
        df_res = pd.DataFrame(columns=['Time', 'Side', 'Volume', 'Impact', 'BID', 'ASK'])
        T, deltas = self.generate_order_times(intensity, start_time, end_time)
        bids, asks = self.generate_bid_ask_prices(deltas, start_price)
        side, volumes, impacts = self.generate_market_orders(deltas.shape[0], buy_share, Vx0_share)

        df_res['Time'] = T
        df_res['Side'] = side
        df_res['Volume'] = volumes
        df_res['Impact'] = impacts
        df_res['BID'] = bids
        df_res['ASK'] = asks
        return df_res