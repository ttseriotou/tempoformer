import gc
gc.collect()

import torch
torch.cuda.empty_cache()

import sys
import numpy as np
from datetime import date
import pickle

sys.path.append('../../data/')


from topicshift_processing import (
    df,
    text_col,
    label_col,
    timeline_col,
)

sys.path.append('../../')
from utils.data_utils import DataFormat

padding = DataFormat(w=20,col_by=timeline_col)
text_pad = padding.pad_np(df[text_col], df)
print(text_pad.shape)

import pickle

folds_fname = '/import/nlp/tt003/topic-shift-mi/data/folds/topicshift_timelinesplit_5fold.pkl'
with open(folds_fname, 'rb') as f:
    fold_list = pickle.load(f)
len(fold_list)

import torch
from datetime import date
from utils.hyper_tuning import Tuning

params = {}
params['MAX_LEN'] = 512
params['batch_size'] = 1
params['num_epochs'] = 4
params['window'] = 20
params['patience'] = 3
params['gamma'] = 2
params['model_type'] = 'roseqbert'
params['output_dim'] = 2
params['num_warmup_steps'] = 0
params['gradient_acc'] = 3
params['dropout_ffn'] = 0.1
params['pos_word_embedding']= 'learnable'
params['sequential_pooling']= 'mha_rotary'
params['connection'] = 'gatednorm' #'gatednorm'  #[None, 'layernorm', 'gatednorm']
params['position_seqlayer'] = [10,11]
params['hidden_dim_ffn'] = [30,30]
params['lstm_hidden_dim']=100
params['loss']='focal'

params['dirname'] = '/import/nlp/tt003/seq-transformer/results/v3/sanctus/'
params['file_name_results'] = 'topicshift_rotempoformer_gating-20posts_5fold_seeds_tuning_'
params['run_date'] = date.today().strftime("%d-%m-%Y")
file_name_results = params['dirname'] + params['file_name_results'] + date.today().strftime("%d-%m-%Y") + '.pkl'
params['save_model'] = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:1'
params['device'] = device

learning_r = [5e-6]
seeds_list = [0, 1, 12,123]

results = []

for lr in learning_r:
    params['learning_rate'] = lr
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
        x_data['reps'] = text_pad
        x_data['id'] = df.index
        results_t = tuning.tune(params,df.copy(), x_data)
        print(results_t[0]['params'])

        results.append(results_t)
        pickle.dump(results, open(file_name_results, 'wb'))
