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

cols = ['coin', 'model', 'start_train', 'end_train', 'test_size', 'MSE', 'scaler']

start_train = pd.to_datetime('2019-01-01').date()
end_train = pd.to_datetime('2022-08-01').date()

exp = pd.DataFrame(columns = cols)

for coin in coins['coin'].values:
    df = pd.read_sql_query('select * from ' + coin, con)
    history_from_sql(df)
    target = 'Close'
    features = list(df.columns)
    train_size = int(len(df)*.8)
    train_df = df[:train_size+7]
    test_df = df[train_size-21:]
    scaled_train_df, scalers = scale_df(train_df)
    scaled_test_df = apply_scalers(test_df, scalers)
    train_dataset = SequenceDataset(scaled_train_df, target, features)
    test_dataset = SequenceDataset(scaled_test_df, target, features)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    n_epochs = 5
    learning_rate = 0.001
    loss_fn = torch.nn.MSELoss()
    
    lstm = LSTM()
    transformer = Transformer()
    dlinear = DLinear()
    
    optimiser1 = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    optimiser2 = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
    optimiser3 = torch.optim.Adam(dlinear.parameters(), lr=learning_rate)
    
    train(n_epochs=n_epochs, model=lstm, optimiser=optimiser1, loss_fn=loss_fn, train_loader=train_loader)
    train(n_epochs=n_epochs, model=transformer, optimiser=optimiser2, loss_fn=loss_fn, train_loader=train_loader, model_type=1)
    train(n_epochs=n_epochs, model=dlinear, optimiser=optimiser3, loss_fn=loss_fn, train_loader=train_loader)
    
    preds1, loss1 = test(lstm, loss_fn, test_loader)
    preds2, loss2 = test(transformer, loss_fn, test_loader, model_type=1)
    preds3, loss3 = test(dlinear, loss_fn, test_loader)
    
    temp1 = [[coin, 'LSTM', start_train, end_train, 0.8, loss1, 'MinMaxScaler(0,1)']]
    temp2 = [[coin, 'Transformer', start_train, end_train, 0.8, loss2, 'MinMaxScaler(0,1)']]
    temp3 = [[coin, 'DLinear', start_train, end_train, 0.8, loss3, 'MinMaxScaler(0,1)']]
    exp = pd.concat((exp, pd.DataFrame(temp1, columns = cols)))
    exp = pd.concat((exp, pd.DataFrame(temp2, columns = cols)))
    exp = pd.concat((exp, pd.DataFrame(temp3, columns = cols)))
    
    scaled_df, scalers = scale_df(df)
    dataset = PredDataset(scaled_df, target, features)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    lstm = LSTM()
    transformer = Transformer()
    dlinear = DLinear()
    optimiser1 = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    optimiser2 = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
    optimiser3 = torch.optim.Adam(dlinear.parameters(), lr=learning_rate)
    train(n_epochs=n_epochs, model=lstm, optimiser=optimiser1, loss_fn=loss_fn, train_loader=loader)
    train(n_epochs=n_epochs, model=transformer, optimiser=optimiser2, loss_fn=loss_fn, train_loader=loader, model_type=1)
    train(n_epochs=n_epochs, model=dlinear, optimiser=optimiser3, loss_fn=loss_fn, train_loader=loader)
    
    con.execute('update coins set last_trained_L=? where coin=?', (end_train, coin))
    con.execute('update coins set last_trained_T=? where coin=?', (end_train, coin))
    con.execute('update coins set last_trained_D=? where coin=?', (end_train, coin))
    
    torch.save(lstm,'models/' + coin + '_lstm')
    torch.save(transformer,'models/' + coin + '_transformer')
    torch.save(dlinear,'models/' + coin + '_dlinear')

exp.to_sql('training_logs', con, if_exists='replace')
con.commit()
con.close()




