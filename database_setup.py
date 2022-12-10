import pandas as pd
import sqlite3 as db
from yfapi import YahooFinanceAPI, Interval
import datetime
import utils

con = db.connect('database.db')

coins = ['BTC', 'ETH', 'BNB', 'XRP', 'DOGE', 'LTC', 'SOL', 'SHIB', 'MATIC']

now = pd.to_datetime('2022-08-01').date()
then = pd.to_datetime('2019-01-01').date()

coins_df  = pd.DataFrame()

coins_df['coin'] = coins

for s in ['L', 'T', 'D']:
    coins_df['last_predicted_' + s] = now
    coins_df['first_trained_' + s] = then
    coins_df['last_trained_' + s] = now

dh = YahooFinanceAPI(Interval.DAILY)

for coin in coins:
    s = coin + '-USD'
    df = dh.get_ticker_data(s, then, now)
    if df.isna().sum().sum() == 0:
        utils.preprocess_data(df)
        df.reset_index()
        df.to_sql(coin, con, if_exists='replace')
    else:
        coin_df.drop(coin, axis=0, inplace=True)

coins_df.to_sql('coins', con, if_exists='replace')

con.close()





