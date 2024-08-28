from __future__ import annotations
import torch
import numpy as np
import pandas as pd
import re
from typing import Optional
from typing import Iterable
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.word_embeddings import SeqDataset
from transformers import BertTokenizer, RobertaTokenizer

class DataFormat:
    def __init__(self, 
                w: int, 
                col_by: str):

        self.pad_window = w
        self.pad_with = 0
        self.col_by = col_by
        
    def pad(self, X, df, pad_before=True):    
        #iterate to create slices
        start_i = 0
        end_i = 0
        sample_list = []
        dim_add = 0

        while X.dim()<3:
            dim_add += 1
            X = X.unsqueeze(-1)

        zeros = torch.zeros((1, X.shape[1], X.shape[2]))

        for i in range(df.shape[0]):
            if (i==0):
                i_prev = 0
            else:
                i_prev = i-1
            if (df[self.col_by][i]==df[self.col_by][i_prev]):
                end_i +=1
                if ((end_i - start_i) > self.pad_window):
                    start_i = end_i - self.pad_window
            else: 
                start_i = i
                end_i = i+1

            #data point with history
            x_add = X[start_i:end_i][np.newaxis, :, :]
            #padding length
            padding_n = self.pad_window - (end_i- start_i) 
            
            #create padding
            zeros_tile = zeros.repeat(padding_n,1,1)
            zeros_tile = zeros_tile.unsqueeze(0)
            #append zero padding
            if pad_before:
                x_padi = torch.cat((zeros_tile, x_add), dim=1)
            else:
                x_padi = torch.cat((x_add, zeros_tile), dim=1)
            #append each sample to final list
            sample_list.append(x_padi)
        
        sample_list = torch.cat((sample_list), dim=0)

        while dim_add>0:
            sample_list = sample_list.squeeze(-1)
            dim_add -=1
        return sample_list

    def pad_np(self, X, df, pad_before=True):
        #iterate to create slices
        start_i = 0
        end_i = 0
        sample_list = []

        zeros = np.zeros((1))

        for i in range(df.shape[0]):
            if (i==0):
                i_prev = 0
            else:
                i_prev = i-1
            if (df[self.col_by][i]==df[self.col_by][i_prev]):
                end_i +=1
                if ((end_i - start_i) > self.pad_window):
                    start_i = end_i - self.pad_window
            else: 
                start_i = i
                end_i = i+1

            #data point with history
            x_add = X[start_i:end_i][np.newaxis, :]
            #padding length
            padding_n = self.pad_window - (end_i- start_i) 
                    
            #create padding
            zeros_tile = np.repeat(zeros, repeats=padding_n)
            zeros_tile = zeros_tile[np.newaxis, :]
            #append zero padding
            if pad_before:
                x_padi = np.concatenate((zeros_tile, x_add), axis=1)
            else:
                x_padi = np.concatenate((x_add, zeros_tile), axis=1)
            #append each sample to final list
            sample_list.append(x_padi)

        sample_list = np.concatenate((sample_list), axis=0)  
        return sample_list
  

    def get_current_embedding(self, df):
        bert_embeddings = torch.tensor(df[[c for c in df.columns if re.match("^e\w*[0-9]", c)]].values)
        return bert_embeddings


class DataLabels:
    def __init__(self):
        #dictionary of labels - client
        y_dict3 = {}
        y_dict3['neutral'] = 0
        y_dict3['sustain'] = 1
        y_dict3['change'] = 2
        self.y_dict3 = y_dict3

        #dictionary of labels - therapist
        y_dict4 = {}
        y_dict4['therapist_input'] = 0
        y_dict4['reflection'] = 1
        y_dict4['question'] = 2
        y_dict4['other'] = 3
        self.y_dict4 = y_dict4

    def get_labels_client(self, df, label_colname):
        #get the flat y labels
        y_data = df[label_colname].values
        y_data = np.array([4 if pd.isnull(xi) else self.y_dict3[xi] for xi in y_data])
        y_data = torch.from_numpy(y_data.astype(int))
        
        return y_data

    def get_labels_therapist(self, df, label_colname):
        #get the flat y labels
        y_data = df[label_colname].values
        y_data = np.array([4 if pd.isnull(xi) else self.y_dict4[xi] for xi in y_data])
        y_data = torch.from_numpy(y_data.astype(int))
        
        return y_data

class DataSplits:
    def __init__(self, col_by, col_y, indices_remove=None, test_split=0.2, data_split_seed=0):
        self.data_split_seed = data_split_seed
        self.splitter_test = GroupShuffleSplit(test_size=test_split, random_state=self.data_split_seed)
        self.col_by = col_by
        self.col_y = col_y
        self.indices_remove = indices_remove

    def get_test_splits(self, df, x_d):
        X = {}
        X['test'] = {}
        X['train'] = {}
        y = {}
        #remove indices
        if (self.indices_remove==None):
            split = self.splitter_test.split(df, groups=df[self.col_by])
        else:
            df = df[df[self.col_y]!=self.indices_remove]
            keep_ind = df.index
            split = self.splitter_test.split(df, groups=df[self.col_by])
            for x_set in x_d.keys():
                x_d[x_set] = x_d[x_set][keep_ind]
            df = df.reset_index(drop=True)

        #train/test 
        train_inds, test_inds = next(split)
        df_train = df.iloc[train_inds]
        df_test = df.iloc[test_inds]
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        #y data
        y['test'] = torch.tensor(df_test[self.col_y].values)
        y['train'] = torch.tensor(df_train[self.col_y].values)
        #x data
        for x_set in x_d.keys():
            X['test'][x_set] = x_d[x_set][test_inds]
            X['train'][x_set] = x_d[x_set][train_inds]
        
        return X, y, df_train

    def get_valid_split(self, df_train, x_d, labels, dev_split=0.25, device=None):
        splitter_dev = GroupShuffleSplit(test_size=dev_split, random_state=self.data_split_seed)
        split = splitter_dev.split(df_train, groups=df_train[self.col_by])
        train_inds, dev_inds = next(split)

        Xk = {}
        Xk['dev'] = {}
        Xk['train'] = {}
        yk = {}
        #train/dev
        for x_set in x_d.keys():
            if torch.is_tensor(x_d[x_set]):
                x_d[x_set] = x_d[x_set].to(device)
            Xk['dev'][x_set] = x_d[x_set][dev_inds]
            Xk['train'][x_set] = x_d[x_set][train_inds]
        yk['dev'] = labels[dev_inds].to(device)
        yk['train'] = labels[train_inds].to(device)

        return Xk, yk
    
    def get_split(self, split_ind, dfk, x_dk, device=None):
        X = {}
        Xk = {}
        X['test'] = {}
        Xk['train'] = {}
        Xk['dev'] = {}
        y = {}
        yk = {}
        train_inds = split_ind[0][0]
        test_inds = split_ind[0][1]
        dev_inds = split_ind[0][2]
 
        df_train = dfk.iloc[train_inds]
        df_test = dfk.iloc[test_inds]
        df_dev = dfk.iloc[dev_inds]
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        df_dev = df_dev.reset_index(drop=True)
        #y data
        y['test'] = torch.tensor(df_test[self.col_y].values).to(device)
        yk['train'] = torch.tensor(df_train[self.col_y].values).to(device)
        yk['dev'] = torch.tensor(df_dev[self.col_y].values).to(device)
        #x data
        for x_set in x_dk.keys():
            if torch.is_tensor(x_dk[x_set]):
                x_dk[x_set] = x_dk[x_set].to(device)
            X['test'][x_set] = x_dk[x_set][test_inds]
            Xk['train'][x_set] = x_dk[x_set][train_inds]
            Xk['dev'][x_set] = x_dk[x_set][dev_inds]

        return Xk, yk, X, y
 
class KFold:
    def __init__(self, n_splits, col_by, col_y, indices_remove=None, data_split_seed=0):
        self.splits = n_splits
        self.col_by = col_by
        self.col_y = col_y
        self.indices_remove = indices_remove
        self.data_split_seed = data_split_seed
        self.splitter_kfold = GroupKFold(n_splits)

    def _get_folds_inds(self, df, x_d):
        #remove indices
        if (self.indices_remove==None):
            #train/dev
            fold_indices = list(self.splitter_kfold.split(df, groups=df[self.col_by]))
        else:
            df = df[df[self.col_y]!=self.indices_remove]
            keep_ind = df.index
            fold_indices = list(self.splitter_kfold.split(df, groups=df[self.col_by]))
            for x_set in x_d.keys():
                x_d[x_set] = x_d[x_set][keep_ind]
            df = df.reset_index(drop=True)

        return fold_indices, df, x_d

    def get_k_fold(self, fold_indices, fold, dfk, x_dk, device=None):
        X = {}
        X['train'] = {}
        X['test'] = {}
        X['dev'] = {}
        y = {}
        train_inds = fold_indices[fold][0]
        test_inds = fold_indices[fold][1]
        if (len(fold_indices[fold])>2) :
            dev_inds = fold_indices[fold][2]
        else:
            train_inds, dev_inds = train_test_split(
                            train_inds,
                            test_size=0.3,
                            random_state=self.data_split_seed,
                        )
        df_train = dfk.iloc[train_inds]
        df_test = dfk.iloc[test_inds]
        df_dev = dfk.iloc[dev_inds]
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        df_dev = df_dev.reset_index(drop=True)
        #y data
        y['train'] = torch.tensor(df_train[self.col_y].values).to(device)
        y['test'] = torch.tensor(df_test[self.col_y].values).to(device)
        y['dev'] = torch.tensor(df_dev[self.col_y].values).to(device)
        #x data
        for x_set in x_dk.keys():
            if torch.is_tensor(x_dk[x_set]):
                x_dk[x_set] = x_dk[x_set].to(device)
            X['train'][x_set] = x_dk[x_set][train_inds]
            X['test'][x_set] = x_dk[x_set][test_inds]
            X['dev'][x_set] = x_dk[x_set][dev_inds]

        return X, y

class Loader:
    def __init__(self, BATCH_SIZE=32, SHUFFLE=True):
        self.BATCH_SIZE = BATCH_SIZE
        self.SHUFFLE = SHUFFLE

    def get_loader(self, X_set, y_set):
        split_set = TensorDataset(*list(X_set.values()), y_set)
        load = DataLoader(dataset=split_set, batch_size = self.BATCH_SIZE, shuffle = self.SHUFFLE)
        return load

class GetSplits:
    def __init__(self, 
        n_folds: int, 
        col_by: str,  
        col_y: str,  
        model_type: str, 
        indices_remove: Optional[int] = None, 
        split_indices: Optional[tuple[Iterable[int]]] = None,
        batch_size: int = 16, 
        max_len: Optional[str] = None,
        device: Optional[str] = None):

        self.n_folds = n_folds
        self.indices_remove = indices_remove
        self.col_by = col_by
        self.col_y = col_y
        self.split_indices = split_indices
        self.batch_size = batch_size
        self.model_type = model_type
        self.max_len = max_len
        self.device = None if (('bert' in self.model_type)) else device

        self.load = Loader(batch_size)

    def splitting(self, dframe, xd, k):
        #loaders
        loader = {}
        #kfold
        if (self.n_folds > 1) :
            kfold = KFold(self.n_folds, self.col_by, self.col_y, self.indices_remove)
            if (self.split_indices==None):
                fold_indices, df, xd = kfold._get_folds_inds(dframe, xd)
                Xk, yk = kfold.get_k_fold(fold_indices, k, df, xd, self.device)
            else:
                Xk, yk = kfold.get_k_fold(self.split_indices, k, dframe, xd, self.device)
            if (('bert' in self.model_type)):
                #tokenizer
                if ('roberta' in self.model_type):
                    tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
                elif ('bert' in self.model_type):
                    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                #data loaders
                par = {'batch_size': self.batch_size,'shuffle': True,'num_workers': 32}
                if ('seq' in self.model_type) & ('time_diff' in Xk['train'].keys()):
                    training_set = SeqDataset(Xk['train']['reps'], yk['train'], tokenizer, self.max_len, Xk['train']['time_diff'], Xk['train']['id'])
                    dev_set = SeqDataset(Xk['dev']['reps'], yk['dev'], tokenizer, self.max_len, Xk['dev']['time_diff'], Xk['dev']['id'])
                    test_set = SeqDataset(Xk['test']['reps'], yk['test'], tokenizer, self.max_len, Xk['test']['time_diff'], Xk['test']['id'])
                elif ('seq' in self.model_type):
                    training_set = SeqDataset(Xk['train']['reps'], yk['train'], tokenizer, self.max_len, id=Xk['train']['id'])
                    dev_set = SeqDataset(Xk['dev']['reps'], yk['dev'], tokenizer, self.max_len, id=Xk['dev']['id'])
                    test_set = SeqDataset(Xk['test']['reps'], yk['test'], tokenizer, self.max_len, id=Xk['test']['id'])
                loader['train'] = DataLoader(training_set, **par)
                loader['dev'] = DataLoader(dev_set, **par)
                loader['test'] = DataLoader(test_set, **par)
            else:
                loader['test'] = self.load.get_loader(Xk['test'], yk['test'])
                loader['train'] = self.load.get_loader(Xk['train'], yk['train'])
                loader['dev'] = self.load.get_loader(Xk['dev'], yk['dev'])
            
            del Xk
        else:
            #get train/test data splits
            datasplits = DataSplits(col_by=self.col_by, col_y=self.col_y, indices_remove=self.indices_remove)
            if (self.split_indices==None):
                X, y,df_train = datasplits.get_test_splits(dframe, xd)
                Xk, yk = datasplits.get_valid_split(df_train, X['train'], y['train'], self.device)
            else:
                Xk, yk, X, y = datasplits.get_split(self.split_indices, dframe, xd, self.device)
            if (('bert' in self.model_type)):
                #tokenizer
                if ('roberta' in self.model_type):
                    tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
                elif ('bert' in self.model_type):
                    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                #data loaders
                par = {'batch_size': self.batch_size,'shuffle': True,'num_workers': 32}
                if ('seq' in self.model_type) & ('time_diff' in Xk['train'].keys()):
                    training_set = SeqDataset(Xk['train']['reps'], yk['train'], tokenizer, self.max_len, Xk['train']['time_diff'], Xk['train']['id'])
                    dev_set = SeqDataset(Xk['dev']['reps'], yk['dev'], tokenizer, self.max_len, Xk['dev']['time_diff'], Xk['dev']['id'])
                    test_set = SeqDataset(X['test']['reps'], y['test'], tokenizer, self.max_len, X['test']['time_diff'], X['test']['id'])
                elif ('seq' in self.model_type):
                    training_set = SeqDataset(Xk['train']['reps'], yk['train'], tokenizer, self.max_len, id=Xk['train']['id'])
                    dev_set = SeqDataset(Xk['dev']['reps'], yk['dev'], tokenizer, self.max_len, id=Xk['dev']['id'])
                    test_set = SeqDataset(X['test']['reps'], y['test'], tokenizer, self.max_len, id=X['test']['id'])
                loader['train'] = DataLoader(training_set, **par)
                loader['dev'] = DataLoader(dev_set, **par)
                loader['test'] = DataLoader(test_set, **par)
            else:
                loader['test'] = self.load.get_loader(X['test'], y['test'])
                loader['train'] = self.load.get_loader(Xk['train'], yk['train'])
                loader['dev'] = self.load.get_loader(Xk['dev'], yk['dev'])
        
            del Xk, X
        return loader, yk



    