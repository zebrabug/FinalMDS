import datetime

import numpy as np
import numba
import pandas as pd


class BaseStrategy:
    """
    Base class for any strategy.
    Contains functions to convert dataframes to numpy arrays (for numba) and vice versa
    """
    def __init__(self, ticksize=0., basis='hour'):
        self.ticksize = ticksize
        self.basis = basis
        if basis == 'hour':
            self.base = 3600
        elif basis == 'minute':
            self.base = 60
        else:
            raise ValueError('Not supported basis')

    def df2np(self, df, columns=None):
        """
        Market order book dataframe -> np.ndarray
        :param df: dataframe,
        :param columns: dict-like, correspond columns (Time, Side, Volume, Impact, BID, ASK) to df columns
        :return: ndarray following structure:
         0     1     2       3       4    5
         Time, Side, Volume, Impact, Bid, Ask
        """
        res = np.zeros(shape=(df.shape[0], 6))
        if columns is None:
            res[:, 0] = df['Time'].map(lambda t: (t - df['Time'].iloc[0]).total_seconds() / self.base)
            res[:, 1] = df['Side']
            res[:, 2] = df['Volume']
            res[:, 3] = df['Impact']
            res[:, 4] = df['BID']
            res[:, 5] = df['ASK']
        else:
            res[:, 0] = df[columns['Time']].map(lambda t: (t - df[columns['Time']].iloc[0]).total_seconds() / self.base)
            res[:, 1] = df[columns['Side']]
            res[:, 2] = df[columns['Volume']]
            res[:, 3] = df[columns['Impact']]
            res[:, 4] = df[columns['BID']]
            res[:, 5] = df[columns['ASK']]

        return res

    def np2df(self, deals, start_time, columns=None, verbose=False):
        """
        Market-maker deals np.ndarray -> dataframe
        :param deals: array of deals
        :param start_time: base start datetime
        :param columns: columns names for DataFrame, 10 or 15 (verbose=True)
        :param verbose: extended DataFrame with info about market orders
        :return: dataframe
        """
        if columns is None:
            if verbose:
                columns = ['Time', 'Wealth', 'Inventory', 'Side', 'Mid', 'Bid', 'Ask', 'R-price', 'Spread', 'PnL',
                           'moImpact', 'moVolume', 'moSide', 'moBID', 'moASK']
            else:
                columns = ['Time', 'Wealth', 'Inventory', 'Side', 'Mid', 'Bid', 'Ask', 'R-price', 'Spread', 'PnL']
        df = pd.DataFrame(columns=columns)

        for i, c in enumerate(columns):
            df.loc[:, c] = deals[:, i]

        if self.basis == 'hour':
            df['Time'] = df['Time'].map(lambda t: start_time + datetime.timedelta(hours=t))
        else:
            df['Time'] = df['Time'].map(lambda t: start_time + datetime.timedelta(minutes=t))
        return df


# ==============================================================================================================
# ========== EQUIDISTANT STRATEGY ===============

@numba.njit(fastmath=True)
def _run_period_EquiDist_adj(orders, W, Q, distance, ticksize=0., strict=False, verbose=True):
    """
    Advanced equidistant strategy with one-tick spread shift
    Return numpy array with deals
    :param orders: input array of market orders
        0     1     2       3       4    5
        Time, Side, Volume, Impact, Bid, Ask
    :param W: start wealth
    :param Q: start inventory
    :param distance: spread/2 in ticks or absolute (no ticksize)
    :param ticksize: 0. for continuous prices
    :return: numpy array of deals
    Array structure:
        0     1       2          3          4    5    6    7        8        9
        Time, Wealth, Inventory, Deal_side, Mid, Bid, Ask, R-price, Spread, PnL
    """
    T = orders[-1, 0]  # time horizon
    deals = np.empty((1, 15 if verbose else 10))  # empty result array with 1 raw

    for time, side, vol, imp, bid, ask in orders:
        mid = (bid + ask) / 2

        # PRICES ROUNDING
        if ticksize != 0:
            # Adjustment
            if round((ask-bid)/ticksize % 2) == 1.:  # odd asymmetric spread
                if Q > 0:  # shift mid to bid to buy
                    mm_bid = mm_ask = mid - ticksize/2
                elif Q < 0:  # shift mid to ask to sell
                    mm_bid = mm_ask = mid + ticksize/2
                else:  # Q == 0  # widen spread
                    mm_bid = mid - ticksize/2
                    mm_ask = mid + ticksize/2
            else:  # even symmetric spread
                mm_bid = mm_ask = mid

            # add distance
            mm_bid -= distance * ticksize
            mm_ask += distance * ticksize

            # order should not be better then best_bid/best_ask
            # mm_bid = bid - max(0., bid - mm_bid)  # not better then best_bid
            # mm_ask = ask + max(0., mm_ask - ask)  # not better then best ask

        else:
            mm_bid, mm_ask = mid - distance, mid + distance

        delta_bid, delta_ask = bid - mm_bid, mm_ask - ask
        deal_side = 0
        # Execute orders
        if side == -1 \
            and (ticksize!=0. and not strict and round((imp-delta_bid)/ticksize) >= 0
                or imp > delta_bid + ticksize/2):
            # bid order execution
            Q += 1
            W -= mm_bid
            deal_side = 1  # I buy

        if side == 1 \
            and (ticksize!=0 and not strict and round((imp-delta_ask)/ticksize) >= 0
                or imp > delta_ask + ticksize/2):
            # ask order execution
            Q -= 1
            W += mm_ask
            deal_side = -1  # I sell

        PnL = Q * mid + W  # PnL uses mid for inventory valuation
        # update dataframe
        if verbose:
            # 10      11      12    13   14
            # impact, volume, side, bid, ask,
            raw = [time, W, Q, deal_side, mid, mm_bid, mm_ask, mid, (mm_ask-mm_bid), PnL, imp, vol, side, bid, ask]
        else:
            raw = [time, W, Q, deal_side, mid, mm_bid, mm_ask, mid, (mm_ask-mm_bid), PnL]
        deals = np.vstack((deals, np.array(raw).reshape(1, -1)))

    return deals[1:]


@numba.njit(fastmath=True)
def _run_period_EquiDist_base(orders, W, Q, distance, ticksize=0., strict=False, verbose=True):
    """
    Simple equidistant strategy.
    Return numpy array with deals
    :param orders: input array of market orders
        0     1     2       3       4    5
        Time, Side, Volume, Impact, Bid, Ask
    :param W: start wealth
    :param Q: start inventory
    :param distance: spread/2 in ticks or absolute (no ticksize)
    :param ticksize: 0. for continuous prices
    :return: numpy array of deals
    Array structure:
        0     1       2          3          4    5    6    7        8        9
        Time, Wealth, Inventory, Deal_side, Mid, Bid, Ask, R-price, Spread, PnL
    """
    T = orders[-1, 0]  # time horizon
    deals = np.empty((1, 15 if verbose else 10))  # empty result array with 1 raw

    for time, side, vol, imp, bid, ask in orders:
        mid = (bid + ask) / 2

        # PRICES ROUNDING
        if ticksize != 0:
            if round((ask-bid)/ticksize % 2) == 1.:  # odd asymmetric spread
                mm_bid = mid - ticksize/2
                mm_ask = mid + ticksize/2
            else:  # even symmetric spread
                mm_bid = mm_ask = mid

            # add distance
            mm_bid -= distance * ticksize
            mm_ask += distance * ticksize

            # order should not be better then best_bid/best_ask
            # mm_bid = bid - max(0., bid - mm_bid)  # not better then best_bid
            # mm_ask = ask + max(0., mm_ask - ask)  # not better then best ask

        else:
            mm_bid, mm_ask = mid - distance, mid + distance

        delta_bid, delta_ask = bid - mm_bid, mm_ask - ask
        deal_side = 0
        # Execute orders
        if side == -1 \
            and (ticksize!=0. and not strict and round((imp-delta_bid)/ticksize) >= 0
                or imp > delta_bid + ticksize/2):
            # bid order execution
            Q += 1
            W -= mm_bid
            deal_side = 1  # I buy

        if side == 1 \
            and (ticksize!=0 and not strict and round((imp-delta_ask)/ticksize) >= 0
                or imp > delta_ask + ticksize/2):
            # ask order execution
            Q -= 1
            W += mm_ask
            deal_side = -1  # I sell

        PnL = Q * mid + W  # PnL uses mid for inventory valuation
        # update dataframe
        if verbose:
            # 10      11      12    13   14
            # impact, volume, side, bid, ask,
            raw = [time, W, Q, deal_side, mid, mm_bid, mm_ask, mid, (mm_ask-mm_bid), PnL, imp, vol, side, bid, ask]
        else:
            raw = [time, W, Q, deal_side, mid, mm_bid, mm_ask, mid, (mm_ask-mm_bid), PnL]
        deals = np.vstack((deals, np.array(raw).reshape(1, -1)))

    return deals[1:]


class EquiDistantStrategy(BaseStrategy):
    def __init__(self, ticksize,  basis='hour'):
        super().__init__(ticksize, basis)

    def run_period_sim(self, orders, distance, start_time=None, end_time=None,
                       preprocess=True, adjust_spread=False, strict=False):
        """
        Run simulation
        :param orders:  orders dataframe
        :param distance: distance from mid for orders
        :param start_time, end_time: period for trade
        :param preprocess: for another structure of input dataframe
        :param adjust_spred: use 1-tick shift for spread depends on inventory
        (a bit mor sophisticated strategy than base one)
        :param strict: use strict condition for order execution (impact > distance)
        :return: dataframe with deals
        """
        df = orders
        if start_time:
            df = df[df['Time'] >= start_time]

        if end_time:
            df = df[df['Time'] <= end_time]

        if preprocess:
            df['Volume'] = df['SIZE_sum']
            df['Side'] = (df['AGGRESSOR_SIDE'] == 'B') * 2 - 1

        np_orders = self.df2np(df)
        if adjust_spread:
            np_deals = _run_period_EquiDist_adj(np_orders, W=0, Q=0,
                                distance=distance, ticksize=self.ticksize, strict=strict)
        else:
            np_deals = _run_period_EquiDist_base(np_orders, W=0, Q=0,
                                distance=distance, ticksize=self.ticksize, strict=strict)

        return self.np2df(np_deals, df['Time'].iloc[0])



# ==============================================================================================================
# ========== AS-MODEL STRATEGY ===============

@numba.njit(fastmath=True)
def _get_limit_order(s, t, T, q, mu, sigma, gamma, k):
    """
    Returns A-S model limit order prices
    :param s: current mid
    :param t: current time (as share of unit time horizon)
    :param T: time horizon
    :param q: inventory
    :param mu: market drift
    :param sigma: market std dev
    :param gamma: risk aversion
    :param k: market model parameter
    :return: res-price, spread
    """
    theta_1 = s + mu * (T - t)
    theta_2 = -sigma * sigma * gamma * (T - t)
    spread = -theta_2 + 2/gamma * np.log(1+gamma/k)
    r_price = theta_1 + theta_2 * q

    return r_price, spread


@numba.njit(fastmath=True)
def _run_period_ASmodel1(orders, W, Q, mu, sigma, gamma, k, ticksize=0., min_spread=1, max_dist=0, strict=False, verbose=False):
    """
    Return numpy array with deals.
    :param orders: input array of market orders
        0     1     2       3       4    5
        Time, Side, Volume, Impact, Bid, Ask
    :param W: start wealth
    :param Q: start inventory
    :param mu: market prices drift
    :param sigma: market prices std dev
    :param gamma: risk aversion (gamma > 0)
    :param k: market parameter = alpha / K
    :param ticksize: 0. for continuous prices
    :param min_spread: min spread in ticks
    :param max_dist: max distance from mid(!) in ticks
        works only if ticksize != 0.
    :param strict: Order execution condition Impact > delta_bid/delta_ask, guarantee execution
    :param verbose: add info from market order book to results
    :return: numpy array of deals
    Array structure:
        0     1       2          3          4    5    6    7        8        9
        Time, Wealth, Inventory, Deal_side, Mid, Bid, Ask, R-price, Spread, PnL

    """
    T = orders[-1, 0]  # time horizon
    deals = np.empty((1, 15 if verbose else 10))  # empty result array with 1 raw

    for time, side, vol, imp, bid, ask in orders:
        mid = (bid + ask) / 2
        r_price, spread = _get_limit_order(mid, time, T, Q, mu, sigma, gamma, k)

        # ROUNDING for discrete pricing
        if ticksize != 0.:
            # round spread and r_price
            tickspread = round(spread / ticksize)  # round spread in ticks
            tick_r_price = round(r_price / ticksize)  # round r_price in ticks
            tickspread = max(min_spread, tickspread)  # low spread limit
            if max_dist:
                tickspread = min(tickspread, 2*max_dist)  # high spread limit (if set)

            # check spread size in ticks
            if round(tickspread % 2) == 0:
            # Even spread (symmetric)
                delta_bid = delta_ask = tickspread // 2
            else:
            # Asymmetric spread
                if Q < 0:
                    delta_bid = tickspread // 2
                    delta_ask = tickspread // 2 + 1
                elif Q > 0:
                    delta_bid = tickspread // 2 + 1
                    delta_ask = tickspread // 2
                else:  # Q == 0
                    # Make spread symmetric
                    tickspread += 1  # now it is even
                    delta_bid = delta_ask = tickspread // 2

            # order prices
            mm_bid = (tick_r_price - delta_bid) * ticksize
            mm_ask = (tick_r_price + delta_ask) * ticksize

            # order should not be better then best_bid/best_ask
            # mm_bid = bid - max(0., bid - mm_bid)  # not better then best_bid
            # mm_ask = ask + max(0., mm_ask - ask)  # not better then best ask

        else:
            # estimate order prices without rounding
            mm_bid, mm_ask = r_price - spread / 2, r_price + spread / 2


        deal_side = 0
        delta_bid, delta_ask = bid - mm_bid, mm_ask - ask
        # Execute orders
        if side == -1 \
            and (ticksize!=0. and not strict and round((imp-delta_bid)/ticksize) >= 0
                or imp > delta_bid + ticksize/2):
            # bid order execution
            Q += 1
            W -= mm_bid
            deal_side = 1  # I buy

        if side == 1 \
            and (ticksize!=0 and not strict and round((imp-delta_ask)/ticksize) >= 0
                or imp > delta_ask + ticksize/2):
            # ask order execution
            Q -= 1
            W += mm_ask
            deal_side = -1  # I sell

        PnL = Q * mid + W  # PnL uses mid for inventory valuation
        # update dataframe
        if verbose:
            # 10      11      12    13   14
            # impact, volume, side, bid, ask,
            raw = [time, W, Q, deal_side, mid, mm_bid, mm_ask, r_price, spread, PnL, imp, vol, side, bid, ask]
        else:
            raw = [time, W, Q, deal_side, mid, mm_bid, mm_ask, r_price, spread, PnL]
        deals = np.vstack((deals, np.array(raw).reshape(1, -1)))

    return deals[1:]


class ASModel1Strategy(BaseStrategy):
    def __init__(self, ticksize, gamma, k, sigma, mu=0., basis='hour'):
        """
        Init strategy and set base parameters of the model
        :param ticksize: if 0. then continuous prices are used
        :param gamma: risk aversion
        :param k: model parameter, equals alpha / K
        where alpha is tail exponent of volume distribution
        and K is log(volume) ~ impact regression coefficient
        :param sigma: market prices std for a basis period
        :param mu: market prices drift for a basis period
        :param basis: hour or minute
        """
        super().__init__(ticksize, basis)
        self.gamma = gamma
        self.k = k
        self.mu = mu
        self.sigma = sigma

    def get_order_prices(self, price, time, end_time, inventory):
        """
        Return optimal prices, not used in simulations.
        :param price:
        :param time:
        :param end_time:
        :param inventory:
        :return:
        """
        T = (end_time - time).total_seconds() / self.base
        rprice, spread = _get_limit_order(price, 0, T, inventory, self.mu, self.sigma, self.gamma, self.k)
        return rprice, spread

    def run_period_sim(self, orders, W, Q, min_spread=1, max_distance=0, start_time=None, end_time=None, preprocess=True):
        """
        Return 1 run simulation result
        :param orders:
        :param W:
        :param Q:
        :param min_spread:
        :param max_distance:
        :param start_time:
        :param end_time:
        :param preprocess:
        :return:
        """
        df = orders
        if start_time:
            df = df[df['Time'] >= start_time]

        if end_time:
            df = df[(df['Time'] <= end_time)]

        if preprocess:
            df['Volume'] = df['SIZE_sum']
            df['Side'] = (df['AGGRESSOR_SIDE'] == 'B') * 2 - 1

        np_orders = self.df2np(df)
        np_deals = _run_period_ASmodel1(np_orders, 0, 0, self.mu, self.sigma, self.gamma, self.k,
                                        self.ticksize, min_spread, max_distance)
        return self.np2df(np_deals, df['Time'].iloc[0])



def main():
    pass

if __name__ == '__main__':
    main()
