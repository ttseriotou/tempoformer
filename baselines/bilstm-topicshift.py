#import libraries
import numpy as np
import pandas as pd
import torch
import sys
import pickle
import re

#import data
sys.path.append('../data/')

from topicshift_processing import (
    df,
    text_col,
    label_col,
    timeline_col,
)

#import SBERT embeddings
emb_sbert_filename = '/import/nlp/tt003/topic-shift-mi/data/embeddings/sentence_embeddings.pkl'
#read embeddings
with open(emb_sbert_filename, 'rb') as f:
    sbert_embeddings = pickle.load(f)

#join the two dataframes
embedding_sentence_df = pd.DataFrame(sbert_embeddings, columns = ['e' + str(i+1) for i in range(sbert_embeddings.shape[1])])
df = pd.concat([df.reset_index(drop=True), embedding_sentence_df], axis=1)

#obtain data with history for bilstm
sys.path.append('../')
from utils.data_utils import DataFormat

reps = torch.tensor(df[[c for c in df.columns if re.match("^e\w*[0-9]", c)]].values)

padding = DataFormat(w=20,col_by=timeline_col)
reps_pad = padding.pad(reps, df, pad_before=False)
print(reps_pad.shape)

#read folds
folds_fname = '/import/nlp/tt003/topic-shift-mi/data/folds/topicshift_timelinesplit_5fold.pkl'
with open(folds_fname, 'rb') as f:
    fold_list = pickle.load(f)
len(fold_list)


###########################################################################################
############################HYPERPARAM TUNING##############################################
###########################################################################################
from datetime import date
from utils.hyper_tuning import Tuning

params = {}
params['model_type'] = 'lstm'
params['batch_size'] = 64
params['gamma'] = 2
params['num_epochs'] = 100
params['patience'] = 3
params['output_dim'] = 2
params['window'] = 20
params['gradient_acc'] = 1
params['sbert_embddding_dim'] = 384
params['dropout'] = 0.1
params['bilstm'] = True
params['num_layers'] = 1
params['loss']='focal'

params['dirname'] = '/import/nlp/tt003/seq-transformer/results/v3/sanctus/'
params['file_name_results'] = 'topicshift_bilstm-20posts_5fold_seeds_tuning_'
params['run_date'] = date.today().strftime("%d-%m-%Y")
file_name_results = params['dirname'] + params['file_name_results'] + date.today().strftime("%d-%m-%Y") + '.pkl'
params['save_model'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:2'
params['device'] = device


learning_rate = [1e-3, 5e-4, 1e-4]
hidden_dims = [200, 300, 400]
seeds_list = [0, 1, 12, 123]

results = []

for lr in learning_rate:
    params['learning_rate'] = lr
    for hd in hidden_dims: 
        params['hidden_dims'] = hd
        for s in seeds_list:
            params['seed'] = s
            tuning = Tuning(col_by=timeline_col, 
                col_y=label_col, 
                n_folds=5,
                indices_remove=None,
                split_indices=fold_list,
                device=device,
                random_seed=params['seed'])
            x_data = {}
            x_data['reps'] = reps_pad
            x_data['id'] = torch.tensor(df.index)
            results_t = tuning.tune(params, df.copy(), x_data)
            print(results_t[0]['params'])

            results.append(results_t)
            pickle.dump(results, open(file_name_results, 'wb'))