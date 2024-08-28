import gc
gc.collect()

import torch
torch.cuda.empty_cache()

import numpy as np
import pandas as pd
import torch
import sys
import json
import pickle

sys.path.append('../..')

import time
from datetime import datetime

#read raw data
data_path = "/homes/tt003/seq-transformer-injection/data/conversations.json"
with open(data_path, "r") as f:
    data = json.load(f)



# Convert conversation thread to linear timeline: we use timestamps
# of each post in the twitter thread to obtain a chronologically ordered list.
def tree2timeline(conversation):
    timeline = []
    timeline.append(
        (
            conversation["source"]["id"],
            conversation["source"]["created_at"],
            conversation["source"]["stance"],
            conversation["source"]["text"],
        )
    )
    replies = conversation["replies"]
    replies_idstr = []
    replies_timestamp = []
    for reply in replies:
        replies_idstr.append(
            (reply["id"], reply["created_at"], reply["stance"], reply["text"])
        )
        replies_timestamp.append(reply["created_at"])

    sorted_replies = [x for (y, x) in sorted(zip(replies_timestamp, replies_idstr))]
    timeline.extend(sorted_replies)
    return timeline


stance_timelines = {"dev": [], "train": [], "test": []}
switch_timelines = {"dev": [], "train": [], "test": []}
check = []
count_switch_threads = 0
all_support_switches = 0
all_oppose_switches = 0
count_threads = 0

for subset in list(data.keys()):
    count_threads += len(data[subset])
    for conv in data[subset]:
        timeline = tree2timeline(conv)
        stance_timelines[subset].append(timeline)
        support = 0
        deny = 0
        old_sum = 0
        switch_events = []
        for i, s in enumerate(timeline):
            if s[2] == "support":
                support = support + 1
            elif s[2] == "query" or s[2] == "deny":
                deny = deny + 1

            new_sum = support - deny
            check.append(new_sum)

            if i != 0 and old_sum == 0 and new_sum != 0:
                # A switch in stance from supporting to opposing the claim starts
                if new_sum < 0:
                    switch_events.append((s[0], s[1], -1, s[3]))
                # A switch in stance from opposing to supporting the claim starts
                elif new_sum > 0:
                    switch_events.append((s[0], s[1], 1, s[3]))
            elif (
                i != 0
                and old_sum < 0
                and new_sum < 0
                and -1 in [x[2] for x in switch_events]
            ):
                # A switch in stance from supporting to opposing the claim continues
                switch_events.append((s[0], s[1], -2, s[3]))
            elif (
                i != 0
                and old_sum > 0
                and new_sum > 0
                and 1 in [x[2] for x in switch_events]
            ):
                # A switch in stance from opposing to supporting the claim continues
                switch_events.append((s[0], s[1], 2, s[3]))

            else:
                switch_events.append((s[0], s[1], 0, s[3]))
            old_sum = new_sum

        support_switch = [x[2] for x in switch_events].count(1)
        oppose_switch = [x[2] for x in switch_events].count(-1)

        if support_switch + oppose_switch > 0:
            count_switch_threads = count_switch_threads + 1
            all_support_switches += support_switch
            all_oppose_switches += oppose_switch

        switch_timelines[subset].append(switch_events)

def simplify_label(y):
    # If the label is -2,-1,2 this is is relabeled to 1
    if y != 0:
        y = 1
    return y


for subset in ["train", "dev", "test"]:
    for i, thread in enumerate(switch_timelines[subset]):
        switch_timelines[subset][i] = [
            (x, z, simplify_label(y), u) for (x, z, y, u) in thread
        ]

df_rumours = pd.DataFrame([], columns=["id", "label", "datetime", "text"])

tln_idx = 0
for subset in ["train", "dev", "test"]:
    for e, thread in enumerate(switch_timelines[subset]):
        df_thread = pd.DataFrame(thread)
        df_thread = pd.DataFrame(thread, columns=["id", "datetime", "label", "text"])
        df_thread = df_thread.reindex(columns=["id", "label", "datetime", "text"])

        df_thread["timeline_id"] = str(tln_idx)
        df_thread["set"] = subset
        df_thread["id"] = df_thread["id"].astype("float64")
        df_thread["datetime"] = pd.to_datetime(df_thread["datetime"])
        df_thread["datetime"] = df_thread["datetime"].map(
            lambda t: t.replace(tzinfo=None)
        )
        df_rumours['label'] = df_rumours['label'].astype('int')
        df_rumours = pd.concat([df_rumours, df_thread])
        tln_idx += 1

df_rumours = df_rumours.reset_index(drop=True)


first_record = df_rumours.groupby('timeline_id')['datetime'].min().reset_index().rename(columns={"datetime": "first_post"})
df_rumours = df_rumours.merge(first_record, how='left', on='timeline_id')
df_rumours['timediff'] = df_rumours.apply(lambda x: (x['datetime']-x['first_post']).total_seconds(),axis=1) #/60


from utils.data_utils import DataFormat

padding = DataFormat(w=20,col_by='timeline_id')
text_pad = padding.pad_np(df_rumours.text, df_rumours)
dt_pad = padding.pad(torch.tensor(df_rumours['timediff']), df_rumours)
print(text_pad.shape, dt_pad.shape)

#######READ FOLDS#########
folds_fname = '/import/nlp/tt003/rumours/data/folds/rumours_timelinesplit_5fold.pkl'
with open(folds_fname, 'rb') as f:
    fold_list = pickle.load(f)
print(len(fold_list))

import torch
from datetime import date
from utils.hyper_tuning import Tuning

params = {}
params['MAX_LEN'] = 512
params['batch_size'] = 3
params['num_epochs'] = 4
params['window'] = 20
params['patience'] = 3
params['gamma'] = 2
params['model_type'] = 'seqroberta'
params['output_dim'] = 2
params['num_warmup_steps'] = 0
params['gradient_acc'] = 1
params['dropout_ffn'] = 0.1
params['pos_word_embedding']= 'learnable'
params['sequential_pooling']= 'mha_rotary'
params['connection'] = 'gatednorm' #'gatednorm' #[None, 'layernorm', 'gatednorm']
params['position_seqlayer'] = [10,11]
params['hidden_dim_ffn'] = [64,64]
params['loss']='focal'

params['dirname'] = '/import/nlp/tt003/seq-transformer/results/v3/'
params['file_name_results'] = 'rumours_tempoformer_roberta-20posts_5fold_seeds_tuning_'
params['run_date'] = date.today().strftime("%d-%m-%Y")
file_name_results = params['dirname'] + params['file_name_results'] + date.today().strftime("%d-%m-%Y") + '.pkl'
params['save_model'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:1'
params['device'] = device

learning_r = [1e-5]
seeds_list = [0, 1, 12,123]

results = []

for lr in learning_r:
    params['learning_rate'] = lr
    for s in seeds_list:
        params['seed'] = s
        tuning = Tuning(col_by='timeline_id', 
                col_y='label', 
                n_folds=5,
                indices_remove=None,
                split_indices=fold_list,
                device=device,
                random_seed=params['seed'])
        x_data = {}
        x_data['reps'] = text_pad
        x_data['time_diff'] = dt_pad
        x_data['id'] = df_rumours.index
        results_t = tuning.tune(params,df_rumours.copy(), x_data)
        print(results_t[0]['params'])

        results.append(results_t)
        pickle.dump(results, open(file_name_results, 'wb'))
