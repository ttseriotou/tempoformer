import numpy as np
import pandas as pd
import torch
import sys
from os import listdir
import pickle
import re

sys.path.append('../data/')

from lrs_processing import (
    df,
    text_col,
    label_col,
    timeline_col,
)

sys.path.append('../')


#############################################################################################
##################SBERT EMBEDDINGS###########################################################
#############################################################################################
#read sbert embeddings and concat
import pickle
emb_sbert_filename = '/import/nlp/tt003/rumours/data/embeddings/sentence_embeddings.pkl'
#read embeddings
with open(emb_sbert_filename, 'rb') as f:
    sbert_embeddings = pickle.load(f)

embedding_sentence_df = pd.DataFrame(sbert_embeddings, columns = ['e' + str(i+1) for i in range(sbert_embeddings.shape[1])])
df = pd.concat([df.reset_index(drop=True), embedding_sentence_df], axis=1)


#############################################################################################
##################AVERAGE HISTORY############################################################
#############################################################################################

#assign index for post in the timeline       
df['timeline_index'] = 0
timelineid_list = df[timeline_col].unique().tolist()
first_index = 0
for t_id in timelineid_list:
    t_id_len = len(df[df[timeline_col]==t_id])
    last_index = first_index + t_id_len
    df['timeline_index'][first_index:last_index] = np.arange(t_id_len)
    first_index = last_index

original_size = df.shape[0]

overall_dim = 2 * len([c for c in df.columns if re.match("^e\w*[0-9]", c)])
split_ind = len([c for c in df.columns if re.match("^e\w*[0-9]", c)])
string_ind = "e"

reps = torch.empty((original_size, overall_dim))

for i in range(original_size):
        t_ind = df.loc[i,'timeline_index']
        t_num = df.loc[i,timeline_col]
        reps[i,:split_ind] = torch.tensor(df[(df['timeline_index']< t_ind) & (df[timeline_col]==t_num)][[c for c in df.columns if re.match("^"+string_ind+"\w*[0-9]", c)]].mean(axis =0))
        if (t_ind==0):
            reps[i,:split_ind] = torch.zeros(1, split_ind)
        reps[i,split_ind:] = torch.tensor(df.loc[i, [c for c in df.columns if re.match("^e\w*[0-9]", c)]])

#############################################################################################
##################FOLDS######################################################################
#############################################################################################
#read folds
folds_fname = '/import/nlp/tt003/rumours/data/folds/rumours_timelinesplit_5fold.pkl'
with open(folds_fname, 'rb') as f:
    fold_list = pickle.load(f)
print(len(fold_list))

###########################################################################################
############################HYPERPARAM TUNING##############################################
###########################################################################################
from datetime import date
from utils.hyper_tuning import Tuning

params = {}
params['model_type'] = 'ffn'
params['batch_size'] = 64
params['gamma'] = 2
params['num_epochs'] = 100
params['patience'] = 3
params['output_dim'] = 2
params['sbert_embddding_dim'] = reps.shape[1]
params['gradient_acc'] = 1
params['dropout'] = 0.1
params['loss']='focal'

params['dirname'] = '/import/nlp/tt003/seq-transformer/results/v3/'
params['file_name_results'] = 'rumours_ffnhistory_5fold_seeds_tuning_'
params['run_date'] = date.today().strftime("%d-%m-%Y")
file_name_results = params['dirname'] + params['file_name_results'] + date.today().strftime("%d-%m-%Y") + '.pkl'
params['save_model'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:1'
params['device'] = device

learning_rate = [1e-3, 5e-4, 1e-4]
hidden_dims = [[64, 64], [128, 128], [256, 256], [512, 512]]
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
            x_data['reps'] = reps
            x_data['id'] = torch.tensor(df.index)
            results_t = tuning.tune(params, df.copy(), x_data)
            print(results_t[0]['params'])

            results.append(results_t)
            pickle.dump(results, open(file_name_results, 'wb'))
