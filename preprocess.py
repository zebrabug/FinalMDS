import datetime

import pandas as pd
import numpy as np

import numba


# All preprocessing in one function
def preprocess_deals_data(data, vol_bins=None, vol_labels=None):
    """
    Preprocess dataframe with deals
    Consolidate market orders, returns new dataframe
    Add features: Day of week, Hour, Time
    """
    # aggregate to obtain market order instant impact
    # deals with the same time are caused by one market order
    data = data.groupby(by=['Time', 'AGGRESSOR_SIDE'], as_index=False) \
        .agg({'PRICE': ['min', 'max'], 'SIZE': 'sum', })

    # Flatten columns (!)
    data.columns = ['_'.join(z) if z[1] != '' else z[0] for z in data.columns]

    # Add datetime features
    data['Date'] = data['Time'].map(lambda d: d.date())  # only date
    data['TimeOnly'] = data['Time'].map(lambda d: d.time())  # only time
    data['DOW'] = data['Time'].map(lambda d: d.isoweekday())  # day of week
    data['H'] = data['Time'].map(lambda d: d.hour)  # hour (for filtering)

    # Impact calculation
    data['Impact'] = data['PRICE_max'] - data['PRICE_min']

    return data


@numba.njit()
def process_day(order_time, time, bid, ask):
    """
    Function gets order time array, and market bid, ask and time array
    For each element in order time array it finds bid and ask
    Returns arrays of bids ands asks for orders
    """
    j = 0
    res_bid = np.zeros_like(order_time, dtype=np.float32)
    res_ask = np.zeros_like(order_time, dtype=np.float32)
    for i, t in enumerate(order_time):
        while time[j] < t:
            j += 1
        res_bid[i] = bid[j]
        res_ask[i] = ask[j]

    return res_bid, res_ask


def extract_day(dt, orders, LOB_folder, LOB_prefix):
    """
    Function gets date, opens corresponding file with LOB history
    and return best bid, best ask and time arrays
    Also return order history time array from deals file (all in numpy format for numba)
    """
    orders_time = orders[orders['Date'] == dt.date()]['Time'].to_numpy()

    fname = LOB_folder + LOB_prefix + dt.strftime('%m%d') + '.feather'
    lob_df = pd.read_feather(fname)
    time_lob = lob_df['Time'].to_numpy()
    bid_lob = lob_df['BID_PRICE1'].to_numpy()
    ask_lob = lob_df['ASK_PRICE1'].to_numpy()
    return orders_time, time_lob, bid_lob, ask_lob


def date_range(start_date, end_date, only_workdays=False):
    """
    Simple generator to obtain dates in defined date range
    """
    for i in range(int((end_date - start_date).days) + 1):
        dt = start_date + datetime.timedelta(days=i)
        if only_workdays and dt.isoweekday() in [6, 7]:
            continue
        yield start_date + datetime.timedelta(days=i)


def add_lob_prices(market_orders, start_date, end_date, LOB_folder, LOB_prefix):
    """
    Function processes market order dataframe and add bid and ask columns
    Returns new dataframe
    """
    for dt in date_range(start_date, end_date, only_workdays=True):
        fname = LOB_folder + LOB_prefix + dt.strftime('%m%d') + '.feather'
        try:
            lob_df = pd.read_feather(fname)
        except FileNotFoundError:
            print(f"{dt} skipped")
            continue

        if lob_df.shape[0] == 0:
            print(f"{fname} no quotes")
            continue

        order_time, lob_time, lob_bid, lob_ask = extract_day(dt, market_orders, LOB_folder, LOB_prefix)
        order_bid, order_ask = process_day(order_time, lob_time, lob_bid, lob_ask)

        market_orders.loc[market_orders['Date'] == dt.date(), 'BID'] = order_bid
        market_orders.loc[market_orders['Date'] == dt.date(), 'ASK'] = order_ask

    market_orders['MID'] = (market_orders['BID'] + market_orders['ASK']) / 2
    market_orders['Spread'] = market_orders['ASK'] - market_orders['BID']

    return market_orders


def main(data_file_name, LOB_folder, LOB_prefix, start_date, end_date, output_file_name):
    print('1. Read data')
    data = pd.read_feather(data_file_name)
    print('2. Preprocess data')
    market_orders = preprocess_deals_data(data)
    print('3. Add market quotes')
    market_orders = add_lob_prices(market_orders, start_date, end_date, LOB_folder, LOB_prefix)
    print('4. Save result')
    market_orders.to_feather(output_file_name)
    # print(market_orders[:5])


if __name__ == "__main__":
    main('Data/USDRUB_TOM_trades.feather',
         'Data\\LOB_USDRUB\\', 'LOB_',
         datetime.datetime.strptime('2021-02-01', '%Y-%m-%d'),
         datetime.datetime.strptime('2021-10-06', '%Y-%m-%d'),
         'market_orders_USDRUB_.feather')
    print('Done!')