import numpy as np
import pandas as pd
import numba
from multiprocessing.pool import ThreadPool


@numba.njit(fastmath=True)
def brownian_motion(mu=0., sigma=1., dt=1., steps=1, s=0.0, ticksize=0.):
    """
    Function returns Brownian motion prices
    dS = mu * dt + sigma * sqrt(dt) * E, E - std normal
    :param s: start price
    :param dt: time quant
    :param mu, sigma: Brownian motion parameters
    :param steps: steps ahead for generating prices
    """
    prices = s + np.cumsum(np.ones(steps)*mu*dt + np.random.randn(steps)*sigma*dt**0.5) 
    if ticksize==0.:
        return prices
    else:
        return np.round_(prices/ticksize, 0, prices)*ticksize


@numba.njit(fastmath=True)
def market_order_touch_limit(A, k, delta, dt):
    """
    Check whether order executed or not
    :param A, k: A = L/alpha and k = alpha*K - market parameters in model
    :param delta: distance from mid
    :param dt: time quant
    Returns True if market order happens and touch limit
    """
    prob = A * np.exp(-k * delta) * dt
    return (np.random.rand(1)[0] < prob/2)  # div 2 because use twice for buy and sell


@numba.njit(fastmath=True)
def mm_limit_order(s, t, q, mu, sigma, gamma, k, T):
    """
    Function returns AS-model order r-price and spread
    :param s: current price
    :param t: current time
    :param q: current inventory
    :param mu, sigma:  mid price Brownian motion parameters
    :param k: market parameter (k = alpha*K)
    :param T: time horizon
    :param gamma: risk aversion
    """

    theta_1 = s + mu * (T - t)
    theta_2 = - sigma**2 * gamma * (T - t)
    
    resrv_price = theta_1 + theta_2*q    
    spread = -theta_2 + 2/gamma * np.log(1+gamma/k)
    
    return resrv_price, spread


# Deals DataFrame initialization
def init_df_deals(start_price, start_wealth=0, start_inventory=0):
    """
    Just deals dataframe init
    """
    data = {'Time': [0.],
            'Wealth': [start_wealth],
            'Inventory': [start_inventory],
            'Deal side': [0.],
            'Mid': [start_price],
            'Bid': [start_price],
            'Ask': [start_price],
            'R-price': [start_price],
            'Spread': [0.],
            'PnL': [0.]}
    return pd.DataFrame.from_dict(data)


def add_df_deals(df_deals, np_deals):
    """
    Add deals from numpy.ndarray to result dataframe.
    :param df_deals: dataframe with deals from previous run or empty dataframe
    :param np_deals: numpy.ndarray of deals
    :return: new dataframe with deals
    """
    df_deals = pd.DataFrame(np_deals, columns=df_deals.columns)
    df_deals.loc[:, 'Deal side'] = df_deals['Deal side'].map(int)
    df_deals.loc[:, 'Inventory'] = df_deals['Inventory'].map(int)
    return df_deals[1:]


@numba.njit(fastmath=True)
def simulation_run(np_deals, gamma, A, k, T, dt, mu, sigma, ticksize=0., max_spread=None):
    """
    Run one simulation of model. Function uses only numpy.arrays for numba
    Return numpy array with deals appended to previous one.
    Input deals array structure:
        0     1       2          3          4    5    6    7       8        9
        Time, Wealth, Inventory, Deal_side, Mid, Bid, Ask, R-price, Spread, PnL
    :param gamma: risk aversion
    :param A, K, t, dt, mu, sigma: model parameters
    :param ticksize: size of the tick for rounding
    :param max_spread: limit for maximum spread
    """
    t0 = np_deals[-1, 0]  # last time
    W = np_deals[-1, 1]  # initial wealth (t=0)
    Q = np_deals[-1, 2]  # initial stock inventory
    PnL = np_deals[-1, 9]  # PnL
    mid = np_deals[-1, 4]  # Mid

    n = int(T+dt // dt)
    new_deals = np.empty((1, np_deals.shape[1]))
    for t in np.arange(dt, T+dt, dt):
        mid += mu * dt + np.random.randn() * sigma * dt**0.5  # next mid price

        # limit order parameters
        if ticksize!=0.:
            midr = np.round(mid/ticksize)*ticksize
            r_price, spread = mm_limit_order(midr, t, Q, mu, sigma, gamma, k, T)
            r_price = np.round(r_price/ticksize)*ticksize
            spread = max(2*ticksize, np.round(spread/ticksize)*ticksize)
            if max_spread is not None:
                spread = min(spread, 2*ticksize*max_spread)

        else:
            midr = mid
            r_price, spread = mm_limit_order(midr, t, Q, mu,sigma, gamma, k, T)

        mm_bid, mm_ask = r_price-spread/2, r_price+spread/2
        delta_bid = midr - mm_bid
        delta_ask = mm_ask - midr
        if max_spread is not None:
            delta_bid = min(max_spread*ticksize, max(ticksize, delta_bid))
            delta_ask = min(max_spread*ticksize, max(ticksize, delta_ask))
            mm_bid, mm_ask = midr - delta_bid, midr + delta_ask

        side = 0
        if market_order_touch_limit(A, k, delta_bid, dt):
            # bid order execution
            Q += 1
            W -= mm_bid
            side += 1

        if market_order_touch_limit(A, k, delta_ask, dt):
            # ask order execution
            Q -= 1
            W += mm_ask
            side += 2

        PnL =  Q * mid + W  # PnL uses mid for inventory valuation

        # update dataframe
        raw = [t0+t, W, Q, side, midr, mm_bid, mm_ask, r_price, spread, PnL]
        new_deals = np.vstack((new_deals, np.array(raw).reshape(1, -1)))
        
    return np.vstack((np_deals, new_deals[1:, :]))


@numba.njit(fastmath=True)
def simulation_symm_run(np_deals, spread_start, spread_end, A, k, T, dt, mu, sigma, ticksize=0.):
    """
    Run simulation with simple strategy (linear or constant spread)
    Spread is around the mid price.
    Spread is a linear function of time, in the most simple case constant
    Return numpy array with deals
    Array structure:
        0     1       2          3          4    5    6    7       8        9
        Time, Wealth, Inventory, Deal_side, Mid, Bid, Ask, R-price, Spread, PnL
    :param spread_start, spread_end: spread at t=0 and at t=T
    :param ticksize: size of the tick for rounding
    """
    
    t0 = np_deals[-1, 0]  # last time
    W = np_deals[-1, 1]  # initial wealth (t=0)
    Q = np_deals[-1, 2]  # initial stock inventory
    PnL = np_deals[-1, 9]  # PnL
    mid = np_deals[-1, 4]  # Mid

    new_deals = np.empty((1, np_deals.shape[1]))
    spread = spread_start
    ds = (spread_end - spread_start) / (T//dt)
    for t in np.arange(dt, T+dt, dt):
        mid += mu * dt + np.random.randn() * sigma * dt**0.5  # next mid price
        # limit order parameters
        if ticksize!=0.:
            midr = np.round(mid/ticksize)*ticksize
            spreadr = max(2*ticksize, np.round(spread/ticksize)*ticksize)
        else:
            midr = mid
            spreadr = spread

        mm_bid, mm_ask = midr-spreadr/2, midr+spreadr/2

        side = 0
        if market_order_touch_limit(A, k, spread/2, dt):
            # bid order execution
            Q += 1
            W -= mm_bid
            side += 1

        if market_order_touch_limit(A, k, spread/2, dt):
            # ask order execution
            Q -= 1
            W += mm_ask
            side += 2

        PnL =  Q * mid + W  # PnL uses mid for inventory
        spread += ds

        # update dataframe
        raw = [t0+t, W, Q, side, midr, mm_bid, mm_ask, midr, spreadr, PnL]
        new_deals = np.vstack((new_deals, np.array(raw).reshape(1, -1)))

    return np.vstack((np_deals, new_deals[1:, :]))


def run_sims(n, multicore=1, **kwargs):
    """
    Run several (n) simulations of AS-model and equidistant strategy
    :param n: number of runs
    :param multicore: use multithreading
    :param kwargs: all parameters of simulation_run function:
        start_price, gamma, A, k, T, dt, mu, sigma, ticksize
    :return: dataframe with results
    """
#     Array structure:
#         0     1       2          3          4    5    6    7       8        9
#         Time, Wealth, Inventory, Deal_side, Mid, Bid, Ask, R-price, Spread, PnL

    start_price = kwargs['start_price']
    gamma = kwargs['gamma']
    A, k, T, dt, mu, sigma = kwargs['A'], kwargs['k'], kwargs['T'], kwargs['dt'], kwargs['mu'], kwargs['sigma']
    ticksize = kwargs.get('ticksize', 0.)
    df_init = init_df_deals(start_price)
    res_columns = ['PnL inv', 'Final Q inv', 'PnL const', 'Final Q const']

    func_sim_model = lambda : simulation_run(df_init.to_numpy(), gamma, A, k, T, dt, mu, sigma, ticksize)
    func_sim_symm = lambda spread: simulation_symm_run(df_init.to_numpy(), spread, spread, A, k, T, dt, mu, sigma, ticksize)

    if multicore == 1:
        np_deals = np.array([func_sim_model() for _ in range(n)])
        spreads = np_deals[:, :, 8].mean(axis=1)
        np_deals_symm = np.array([func_sim_symm(s) for s in spreads])

    else:
        with ThreadPool(multicore) as pool:
            np_deals = np.array(pool.starmap(func_sim_model, [() for _ in range(n)], chunksize=n//multicore))
            spreads = np_deals[:, :, 8].mean(axis=1)
            np_deals_symm = np.array(pool.starmap(func_sim_symm, [(s,) for s in spreads], chunksize=n//multicore))

    data = {
        'PnL inv': np_deals[:, -1, 9],
        'Final Q inv': np_deals[:, -1, 2],
        'PnL const': np_deals_symm[:, -1, 9],
        'Final Q const': np_deals_symm[:, -1, 2],
        'Spread inv': np_deals[:, :, 8].mean(axis=1),
        'Spread const': np_deals_symm[:, :, 8].mean(axis=1),
        'Max Q inv': np.abs(np_deals[:, :, 2]).max(axis=1),
        'Max Q const': np.abs(np_deals_symm[:, :, 2]).max(axis=1),
        'N inv': np.where(np_deals[:, :, 3]!=0, 1, 0).sum(axis=1),
        'N const': np.where(np_deals_symm[:, :, 3]!=0, 1, 0).sum(axis=1)
    }

    return pd.DataFrame(data)

# numba multiperiod run functions
@numba.njit(fastmath=True)
def sim_multiperiod_run(periods, np_deals, gamma, A, k, T, dt, mu, sigma, ticksize):
    """
    Run sequence of simulations for Stoikov model
    :param periods: sequence size
    :param np_deals: previous deals array
    :param gamma: risk aversion
    :param A, K, t, dt, mu, sigma: model parameters
    :param ticksize: size of the tick for rounding
    :return: numpy array of the deals
    """
    for _ in range(periods):
        np_deals = simulation_run(np_deals, gamma, A, k, T, dt, mu, sigma, ticksize)
    return np_deals

@numba.njit(fastmath=True)
def sim_multiperiod_symm_run(periods, np_deals, start_spread, end_spread, A, k, T, dt, mu, sigma, ticksize):
    """
    Run sequence of simulations for simple equidistant model
    :param periods: sequence size
    :param np_deals: previous deals array
    :param gamma: risk aversion
    :param A, K, t, dt, mu, sigma: model parameters
    :param ticksize: size of the tick for rounding
    :return: numpy array of the deals
    """
    for _ in range(periods):
        np_deals = simulation_symm_run(np_deals, start_spread, end_spread, A, k, T, dt, mu, sigma, ticksize)
    return np_deals


def run_multiperiod_sims(n, multiperiod, **kwargs):
    """
    Function for multiperiod simulation
    :param n: number of runs
    :param multiperiod: number of periods per run
    :param kwargs: all model parameters
    :return: dataframe with results
    """
#     Array structure:
#         0     1       2          3          4    5    6    7       8        9
#         Time, Wealth, Inventory, Deal_side, Mid, Bid, Ask, R-price, Spread, PnL

    start_price = kwargs['start_price']
    gamma = kwargs['gamma']
    A, k, T, dt, mu, sigma = kwargs['A'], kwargs['k'], kwargs['T'], kwargs['dt'], kwargs['mu'], kwargs['sigma']
    ticksize = kwargs.get('ticksize', 0.)
    np_init = init_df_deals(start_price).to_numpy()
    res_columns = ['PnL inv', 'Final Q inv', 'PnL const', 'Final Q const']

    np_deals = np.array([sim_multiperiod_run(multiperiod, np_init, gamma, A, k, T, dt, mu, sigma, ticksize)
                         for _ in range(n)])
    spreads = np_deals[:, :, 8].mean(axis=1)
    np_deals_symm = np.array([sim_multiperiod_symm_run(multiperiod, np_init, s, s, A, k, T, dt, mu, sigma, ticksize)
                              for s in spreads])

    data = {
        'PnL inv': np_deals[:, -1, 9],
        'Final Q inv': np_deals[:, -1, 2],
        'PnL const': np_deals_symm[:, -1, 9],
        'Final Q const': np_deals_symm[:, -1, 2],
    }

    return pd.DataFrame(data, columns=res_columns)


def main():
    pass

if __name__ == '__main__':
    main()