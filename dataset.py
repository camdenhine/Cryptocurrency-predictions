import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple
from utils import get_residuals

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=21, pred_length=7):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.pred_length = pred_length
        self.Y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0] - self.pred_length - self.sequence_length

    def __getitem__(self, i): 

        x = self.X[i:(i + self.sequence_length), :]
        
        #padding = self.Y[-1].repeat(self.pred_length)
        #y = torch.cat((self.Y,padding), 0)
        i_start = i + self.sequence_length
        i_end = i + self.pred_length + self.sequence_length


        return x, self.Y[i_start-1:i_end-1], self.Y[i_start:i_end]

class PredDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=21, pred_length=7):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.pred_length = pred_length
        self.Y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0] - self.sequence_length

    def __getitem__(self, i): 

        x = self.X[i:(i + self.sequence_length), :]
        
        padding = self.Y[-1].repeat(self.pred_length)
        y = torch.cat((self.Y,padding), 0)
        i_start = i + self.sequence_length
        i_end = i + self.pred_length + self.sequence_length


        return x, y[i_start-1:i_end-1], y[i_start:i_end]

class ResidualSequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=21, pred_length=7):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.pred_length = pred_length
        self.dataframe = dataframe
        self.Y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe.values).float()

    def __len__(self):
        return self.X.shape[0] - self.pred_length - self.sequence_length

    def __getitem__(self, i): 
        
        if i > 50:
            df = self.dataframe[:i + 1 + self.sequence_length].copy()
        else:
            df = self.dataframe[:51 + self.sequence_length].copy()
        df['residuals'] = get_residuals(df['Close'])
        X_r = torch.tensor(df[self.features].values).float()
        x = X_r[i:i + self.sequence_length, :]

        #padding = self.Y[-1].repeat(self.pred_length)
        #y = torch.cat((self.Y,padding), 0)
        i_start = i + self.sequence_length
        i_end = i + self.pred_length + self.sequence_length
        del df

        return x, self.Y[i_start-1:i_end-1], self.Y[i_start:i_end]

class PredResidualSequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=21, pred_length=7):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.pred_length = pred_length
        self.dataframe = dataframe
        self.Y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe.values).float()

    def __len__(self):
        return self.X.shape[0] - self.sequence_length

    def __getitem__(self, i):

        if i > 50:
            df = self.dataframe[:i + 1 + self.sequence_length].copy()
        else:
            df = self.dataframe[:51 + self.sequence_length].copy()

        df['residuals'] = get_residuals(df['Close'])
        X_r = torch.tensor(df[self.features].values).float() 
        x = X_r[i:(i + self.sequence_length), :]

        padding = self.Y[-1].repeat(self.pred_length)
        y = torch.cat((self.Y,padding), 0)
        i_start = i + self.sequence_length
        i_end = i + self.pred_length + self.sequence_length
        del df

        return x, y[i_start-1:i_end-1], y[i_start:i_end]