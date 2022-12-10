from dataset import SequenceDataset, PredDataset
from model import LSTM, Transformer, DLinear
from main import *
from torch.utils.data import DataLoader
from utils import *
import pandas as pd
import torch
import numpy as np
import random
import sqlite3 as db
from yfapi import YahooFinanceAPI, Interval
import datetime
import matplotlib.pyplot as plt

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

con = db.connect('database.db')

coins = pd.read_sql_query('select * from coins', con)

cols = ['Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6', 'Day_7']

for coin in coins['coin'].values:
    df = pd.read_sql_query('select * from ' + coin, con)
    history_from_sql(df)
    target = 'Close'
    features = list(df.columns)
    scaled_df, scalers = scale_df(df)
    
    dh = YahooFinanceAPI(Interval.DAILY)
    now = pd.to_datetime('2022-12-01').date()
    then = pd.to_datetime('2022-08-01').date()
    new_df = dh.get_ticker_data(coin + "-USD", then, now)
    history_from_yahoo(new_df)
    
    new_scaled_df = apply_scalers(new_df, scalers)
    new_scaled_history = pd.concat((scaled_df.iloc[-21:-1], new_scaled_df))
    
    dataset = PredDataset(new_scaled_history, target, features)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    lstm = torch.load('models/' + coin + '_lstm')
    transformer = torch.load('models/' + coin + '_transformer')
    dlinear = torch.load('models/' + coin + '_dlinear')
    loss_fn = torch.nn.MSELoss()
    
    preds1, loss1 = test(lstm, loss_fn, loader)
    preds2, loss2 = test(transformer, loss_fn, loader, model_type=1)
    preds3, loss3 = test(dlinear, loss_fn, loader)
    
    preds1_df = pd.DataFrame(scalers[3].inverse_transform(preds1), columns = cols)
    preds2_df = pd.DataFrame(scalers[3].inverse_transform(preds2), columns = cols)
    preds3_df = pd.DataFrame(scalers[3].inverse_transform(preds3), columns = cols)
    
    preds1_df['Date'] = new_df.iloc[1:].index
    preds2_df['Date'] = new_df.iloc[1:].index
    preds3_df['Date'] = new_df.iloc[1:].index
    
    preds1_df['Date_predicted'] = now
    preds2_df['Date_predicted'] = now
    preds3_df['Date_predicted'] = now
    
    preds1_df['Close'] = new_df['Close'].iloc[1:].reset_index(drop=True)
    preds2_df['Close'] = new_df['Close'].iloc[1:].reset_index(drop=True)
    preds3_df['Close'] = new_df['Close'].iloc[1:].reset_index(drop=True)
    
    preds1_df.to_sql(coin + '_preds_L', con, if_exists='replace')
    preds2_df.to_sql(coin + '_preds_T', con, if_exists='replace')
    preds3_df.to_sql(coin + '_preds_D', con, if_exists='replace')
    
    new_total_df = pd.concat((df.iloc[:-1], new_df))
    new_total_df.reset_index(inplace=True)
    new_total_df.to_sql(coin, con, if_exists='replace')
    
    con.execute('update coins set last_predicted_L=? where coin=?', (now, coin))
    con.execute('update coins set last_predicted_T=? where coin=?', (now, coin))
    con.execute('update coins set last_predicted_D=? where coin=?', (now, coin))

con.commit()
con.close()

