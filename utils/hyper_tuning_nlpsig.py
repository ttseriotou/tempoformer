from __future__ import annotations
import sys
sys.path.append('..')

import torch.nn as nn
import torch
import math
import numpy as np
from typing import Iterable, Optional
from sklearn import metrics
from utils.classification_utils import train, test, validation, set_seed
from utils.loss_functions import FocalLoss
from datetime import datetime
from models.lstm import LSTMModel
from models.seqsignet_bilstm import SeqSigNet
from models.swnu_network import SWNUNetwork
from utils.data_utils import GetSplits
from transformers.optimization import get_linear_schedule_with_warmup

class Tuning:
    def __init__(
        self, 
        col_by: str, 
        col_y: str, 
        n_folds: int, 
        indices_remove: Optional[int] = None,
        split_indices: Optional[tuple[Iterable[int]]] = None,
        device: Optional[str] = None,
        random_seed: int=123):

        self.col_by = col_by
        self.col_y = col_y
        self.indices_remove  = indices_remove
        self.n_folds = n_folds
        self.split_indices = split_indices
        self.device = device
        self.random_seed = random_seed

    def tune(self, parameters, df_tune, x_data_tune):
        set_seed(self.random_seed)
        myGenerator = torch.Generator()
        myGenerator.manual_seed(self.random_seed)

        results_test = []
      
        max_len = parameters['MAX_LEN'] if ('MAX_LEN' in parameters.keys()) else None
        getsplits = GetSplits(self.n_folds, self.col_by, self.col_y, parameters['model_type'],
                    self.indices_remove, self.split_indices, parameters['batch_size'], max_len,
                    device=None)

        labels_t_add_all = torch.empty((0))
        predicted_t_add_all = torch.empty((0))

        for k in range(self.n_folds):
            print('Fold', k)
            loader, yk = getsplits.splitting(df_tune.copy(), x_data_tune.copy(), k)

            #model
            if (parameters['model_type']=='lstm'):
                model = LSTMModel(input_dim = parameters['sbert_embddding_dim'],
                            hidden_dim=parameters['hidden_dims'],
                            num_layers = parameters['num_layers'],
                            bidirectional = parameters['bilstm'],
                            output_dim=parameters['output_dim'],
                            dropout_rate = parameters['dropout']
                    )
            elif (parameters['model_type']=='swnu'):
                model = SWNUNetwork(input_channels=parameters['input_channels'],
                            num_features=parameters['num_features'],
                            embedding_dim=parameters['embedding_dim'],
                            log_signature=parameters['log_signature'],
                            sig_depth=parameters['sig_depth'],
                            pooling=parameters["pooling"],
                            hidden_dim_swnu=parameters['hidden_dim_swnu'],
                            hidden_dim_ffn=parameters['hidden_dim_ffn'],
                            output_dim=parameters['output_dim'],
                            output_channels=parameters['output_channels'],
                            dropout_rate=parameters['dropout_rate'],
                            BiLSTM=parameters['BiLSTM'],
                            comb_method=parameters['comb_method'],
                    )
            elif (parameters['model_type']=='seqsignet'):
                model = SeqSigNet(input_channels=parameters['input_channels'],
                            num_features=parameters['num_features'],
                            embedding_dim=parameters['embedding_dim'],
                            log_signature=parameters['log_signature'],
                            sig_depth=parameters['sig_depth'],
                            pooling=parameters["pooling"],
                            hidden_dim_swnu=parameters['hidden_dim_swnu'],
                            hidden_dim_lstm=parameters['hidden_dim_lstm'],
                            hidden_dim_ffn=parameters['hidden_dim_ffn'],
                            output_dim=parameters['output_dim'],
                            output_channels=parameters['output_channels'],
                            dropout_rate=parameters['dropout_rate'],
                            BiLSTM=parameters['BiLSTM'],
                            comb_method=parameters['comb_method'],
                )
            else:
                raise ValueError("model must be from this list of options: 'seqattn', 'lstm', 'ffn'")
            model.to(self.device) 
            #loss
            y_list = yk['train'].tolist()
            alpha_values = torch.Tensor([math.sqrt(yk['train'].shape[0] / y_list.count(i)) for i in set(y_list)])
            #alpha_values = torch.Tensor([math.sqrt(1/( yk['train'][yk['train']==0].shape[0]/yk['train'].shape[0])), math.sqrt(1/(yk['train'][yk['train']==1].shape[0]/yk['train'].shape[0])), math.sqrt(1/(yk['train'][yk['train']==2].shape[0]/yk['train'].shape[0]))])
            parameters['alpha_values'] = alpha_values 

            if (parameters['loss']=='cross_entropy'):
                criterion = nn.CrossEntropyLoss()                            
            elif (parameters['loss']=='focal') :
                criterion = FocalLoss(gamma = parameters['gamma'], alpha = parameters['alpha_values'])
            if ('num_warmup_steps' in parameters.keys()):
                optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['learning_rate'], eps=1e-08, weight_decay=0)
                parameters['num_training_steps'] = len(loader['train']) *parameters['num_epochs']
                scheduler= get_linear_schedule_with_warmup(optimizer, parameters['num_warmup_steps'], parameters['num_training_steps'])
                #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, parameters['num_training_steps'], eta_min=1e-5)               
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'])  
                scheduler=None

            #early stopping params
            last_metric = 0
            trigger_times = 0
            best_metric = 0
            results_test_fold = {}
            results_test_fold['params'] = {}
            results_test_fold['loss_train'] = []
            results_test_fold['loss_valid'] = []
            results_test_fold['f1_valid'] = []

            #model train/validation per epoch
            for epoch in range(parameters['num_epochs']):
                loss_train = train(model, loader['train'], criterion, optimizer, epoch, parameters['num_epochs'], parameters['gradient_acc'], self.device)

                if (scheduler!=None):
                    scheduler.step()

                # Early stopping
                f1_v, labels_valid, predicted_valid, ids_valid, loss_valid = validation(model, loader['dev'], criterion, parameters['loss'], self.device)
                print('Current Macro F1:', f1_v)

                #append scores per epoch
                results_test_fold['loss_train'].append(loss_train)
                results_test_fold['loss_valid'].append(loss_valid)
                results_test_fold['f1_valid'].append(f1_v)

                if f1_v > best_metric :
                    best_metric = f1_v

                    #test and save so far best model
                    f1_t, labels_test, predicted_test, probs_test, ids_test = test(model, loader['test'], parameters['loss'], self.device)
                    results_test_fold['params'] = parameters.copy()
                    results_test_fold['f1_val'] = f1_v.copy()
                    results_test_fold['f1_test'] = f1_t.copy()
                    results_test_fold['probs_test'] = torch.clone(probs_test)
                    results_test_fold['predicted'] = torch.clone(predicted_test)
                    results_test_fold['labels'] = torch.clone(labels_test)
                    results_test_fold['predicted_valid'] = torch.clone(predicted_valid)
                    results_test_fold['labels_valid'] = torch.clone(labels_valid)
                    results_test_fold['ids_test'] = torch.clone(ids_test)
                    results_test_fold['ids_valid'] = torch.clone(ids_valid)
                    results_test_fold['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    if parameters['save_model']:
                        file_name_model = parameters['dirname'] + 'models/' + parameters['file_name_results'] + str(k) + "fold"  + "_" + str(parameters['seed']) + "seed"  + "_" + parameters['run_date']+'.pkl'
                        torch.save(model.state_dict(), file_name_model)

                if f1_v < last_metric:
                    trigger_times += 1
                    #print('Trigger Times:', trigger_times)

                    if trigger_times >= parameters['patience']:
                        #print('Early stopping!')
                        break
                else:
                    #print('Trigger Times: 0')
                    trigger_times = 0
                last_metric = f1_v
                print('Best so far test F1:', f1_t)
            
            labels_t_add_all = torch.cat([labels_t_add_all, labels_test])
            predicted_t_add_all = torch.cat([predicted_t_add_all, predicted_test])

            #append the best results 
            results_test.append(results_test_fold)

            del loader

        print('F1 Validation for param set:', np.mean([x['f1_val'] for x in results_test]))
        f1_t_add = 100 * metrics.f1_score(labels_t_add_all, predicted_t_add_all, average = 'macro')
        print('F1 Test for param set:', f1_t_add)
        return results_test