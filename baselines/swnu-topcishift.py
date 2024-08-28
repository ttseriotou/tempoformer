import numpy as np
import pandas as pd
import torch
import sys
from os import listdir
import pickle

sys.path.append('../data/')


from topicshift_processing import (
    df,
    text_col,
    label_col,
    timeline_col,
)

sys.path.append('../')

from utils.hyper_tuning_nlpsig import Tuning

sys.path.append('nlpsig-networks/')

#import SBERT embeddings
emb_sbert_filename = '/import/nlp/tt003/topic-shift-mi/data/embeddings/sentence_embeddings.pkl'
#read embeddings
with open(emb_sbert_filename, 'rb') as f:
    sbert_embeddings = pickle.load(f)

from nlpsig_networks.scripts.swnu_network_functions import (
    obtain_SWNUNetwork_input,
)

swnu_input = obtain_SWNUNetwork_input(
    method="umap",
    dimension=15,
    df=df,
    id_column=timeline_col,
    label_column=label_col,
    embeddings=sbert_embeddings,
    k=20,
    features=["timeline_index"],
    standardise_method = [None],
    include_features_in_path=True,
    include_features_in_input=False,
    seed=42,
    path_indices=None,
)

import pickle

folds_fname = '/import/nlp/tt003/topic-shift-mi/data/folds/topicshift_timelinesplit_5fold.pkl'
with open(folds_fname, 'rb') as f:
    fold_list = pickle.load(f)
len(fold_list)

from datetime import date

params = {}
params['model_type'] = 'swnu'
params['batch_size'] = 64
params['gamma'] = 2
params['num_epochs'] = 100
params['patience'] = 3
params['window'] = 20
params['input_channels'] = swnu_input["input_channels"]
params['num_features'] = swnu_input["num_features"]
params['embedding_dim'] = swnu_input["embedding_dim"]
params['log_signature'] = True
params['sig_depth'] = 3
params["pooling"] = "signature"
params['output_dim'] = 2
params['dropout_rate'] = 0.1
params['BiLSTM'] = True
params['comb_method'] = 'concatenation'
params['gradient_acc'] = 1
params['loss']='focal'

params['dirname'] = '/import/nlp/tt003/seq-transformer/results/v3/sanctus/'
params['file_name_results'] = 'topicshift_swnu-20posts_5fold_seeds_tuning_'
params['run_date'] = date.today().strftime("%d-%m-%Y")
file_name_results = params['dirname'] + params['file_name_results'] + date.today().strftime("%d-%m-%Y") + '.pkl'
params['save_model'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:0'
params['device'] = device

learning_rate = [0.0005, 0.0003, 0.0001]
hidden_dims = [[32, 32], [128, 128]]
lstm_swnu = [10,12]
output_channels = [6,8,10]
seeds_list = [0,1,12,123]

results = []

for lr in learning_rate:
    params['learning_rate'] = lr
    for hd in hidden_dims: 
        params['hidden_dim_ffn'] = hd
        for lswnu in lstm_swnu:
            params['hidden_dim_swnu'] = lswnu
            for out_chan in output_channels:
                params['output_channels'] = out_chan
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
                    x_data['path'] = swnu_input['x_data']['path'].float()
                    x_data['features'] = swnu_input['x_data']['features'].float()
                    x_data['id'] = torch.tensor(df.index)
                    results_t = tuning.tune(params, df.copy(), x_data)
                    print(results_t[0]['params'])

                    results.append(results_t)
                    pickle.dump(results, open(file_name_results, 'wb'))