import os
import random
import time

import mlflow
import numpy as np
import pandas as pd

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlflow import log_artifacts, log_metric, log_param
from torch import LongTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaConfig, RobertaModel
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from transformers import AdamW 
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()

from read_candidates import fetchAndTransformCandidates
from Utilities.choosers import *
from Utilities.mlflow_logging import *
from train_eval_predict import *


def convertDf2Tensor(df):
    return torch.from_numpy( np.array( list( df ), dtype=np.int64 )).clone().detach()

def loadModel(model, exp_args):

    if exp_args.parallel == 'true':

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids = [0, 1])
            print("Using", str(model.device_ids), " GPUs!")
            return model

    elif exp_args.parallel == 'false':
        model = nn.DataParallel(model, device_ids = [0])
        return model

if __name__ == "__main__":


    # for eachSeed in [ 0, 1, 42 ]:
    for eachSeed in [ 0 ]:

        with mlflow.start_run() as run:

            def seed_everything( seed ):
                random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
            seed_everything(eachSeed)

            print('The random seed is set to: ', eachSeed)

            # This is executed after the seed is set because it is imperative to have reproducible data run after shuffle
            weak_candidates, ebm_nlp_df,  ebm_gold_df, hilfiker_df, exp_args, current_tokenizer, current_modelembed = fetchAndTransformCandidates()

            if exp_args.train_data == 'distant-cto': 
                fulldf = weak_candidates
            if exp_args.train_data == 'ebm-pico': 
                fulldf = ebm_nlp_df
            if exp_args.train_data == 'combined': 
                fulldf = weak_candidates.append(ebm_nlp_df, ignore_index=True)

            # shuffle the dataset and divide into training and validation sets
            fulldf = fulldf.sample(frac=1).reset_index(drop=True)
            annotations, annotations_devdf_ = train_test_split(fulldf, test_size=0.2) 
            print('Size of training set: ', len(annotations.index))
            print('Size of development set: ', len(annotations_devdf_.index))


            # Log the parameters
            logParams(exp_args, eachSeed)

            # Convert all inputs, labels, and attentions into torch tensors, the required datatype: torch.int64
            train_input_ids = convertDf2Tensor(annotations['embeddings'])
            train_input_labels = convertDf2Tensor(annotations['label_pads'])
            train_attn_masks = convertDf2Tensor(annotations['attn_masks'])
            train_pos_tags = convertDf2Tensor(annotations['inputpos'])
            # train_pos_tags = torch.nn.functional.one_hot( torch.from_numpy( np.array( list(annotations['inputpos']), dtype=np.int64) ).clone().detach() )
            assert train_input_ids.dtype == train_input_labels.dtype == train_attn_masks.dtype == train_pos_tags.dtype

            # Test set (EBM-NLP training data used as test set)
            dev_input_ids = convertDf2Tensor( annotations_devdf_['embeddings'])
            dev_input_labels = convertDf2Tensor( annotations_devdf_['label_pads'])
            dev_attn_masks = convertDf2Tensor( annotations_devdf_['attn_masks'])
            dev_pos_tags = convertDf2Tensor( annotations_devdf_['attn_masks'])
            # test_pos_tags = torch.nn.functional.one_hot( torch.from_numpy( np.array( list(annotations_testdf_['inputpos'])
            assert dev_input_ids.dtype == dev_input_labels.dtype == dev_attn_masks.dtype == dev_pos_tags.dtype

            # Test set 1 (EBM-NLP test gold data test set)
            test1_input_ids = convertDf2Tensor( ebm_gold_df['embeddings'])
            test1_input_labels = convertDf2Tensor( ebm_gold_df['label_pads'])
            test1_attn_masks = convertDf2Tensor( ebm_gold_df['attn_masks'])
            test1_pos_tags = convertDf2Tensor( ebm_gold_df['attn_masks'])

            # Test set 2 (Hilfiker test set)
            test2_input_ids = convertDf2Tensor( hilfiker_df['embeddings'])
            test2_input_labels = convertDf2Tensor( hilfiker_df['label_pads'])
            test2_attn_masks = convertDf2Tensor( hilfiker_df['attn_masks'])
            test2_pos_tags = convertDf2Tensor( hilfiker_df['attn_masks'])

            # ----------------------------------------------------------------------------------------
            # Create dataloaders from the tensors
            # ----------------------------------------------------------------------------------------
            # Create the DataLoader for our training set.
            train_data = TensorDataset(train_input_ids, train_input_labels, train_attn_masks, train_pos_tags)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=None, batch_size=10, shuffle=False)

            # Create the DataLoader for our test set. (This will be used as validation set!)
            dev_data = TensorDataset(dev_input_ids, dev_input_labels, dev_attn_masks, dev_pos_tags)
            dev_sampler = RandomSampler(dev_data)
            dev_dataloader = DataLoader(dev_data, sampler=None, batch_size=10, shuffle=False)

            # Create the DataLoader for our test set 1.
            test1_data = TensorDataset(test1_input_ids, test1_input_labels, test1_attn_masks, test1_pos_tags)
            test1_sampler = RandomSampler(test1_data)
            test1_dataloader = DataLoader(test1_data, sampler=None, batch_size=6, shuffle=False)

            del test1_input_ids, test1_input_labels, test1_attn_masks, test1_pos_tags, test1_sampler
            gc.collect()

            # Create the DataLoader for our test set 2.
            test2_data = TensorDataset(test2_input_ids, test2_input_labels, test2_attn_masks, test2_pos_tags)
            test2_sampler = RandomSampler(test2_data)
            test2_dataloader = DataLoader(test2_data, sampler=None, batch_size=6, shuffle=False)

            del test2_input_ids, test2_input_labels, test2_attn_masks, test2_pos_tags, test2_sampler
            gc.collect()

            ##################################################################################
            #Instantiating the BERT model
            ##################################################################################
            print("Building model...")
            createOSL = time.time()
            model = choose_model(exp_args.embed, current_tokenizer, current_modelembed, exp_args.model, exp_args)

            ##################################################################################
            # Tell pytorch to run data on this model on the GPU and parallelize it
            ##################################################################################

            if exp_args.train_from_scratch == True:
                model = loadModel(model, exp_args)
            else:
                checkpoint = torch.load(exp_args.plugin_model, map_location='cuda:0')
                model.load_state_dict( checkpoint )
                model = loadModel(model, exp_args)
            print('The devices used: ', str(model.device_ids) )

            ##################################################################################
            # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
            optimizer = AdamW(model.parameters(),
                            lr = exp_args.lr, # args.learning_rate - default is 5e-5 (for BERT-base)
                            eps = exp_args.eps, # args.adam_epsilon  - default is 1e-8.
                            )

            # Total number of training steps is number of batches * number of epochs.
            total_steps = len(train_dataloader) * exp_args.max_eps
            print('Total steps per epoch: ', total_steps)

            # Create the learning rate scheduler.
            scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps=total_steps*exp_args.lr_warmup,
                                                        num_training_steps = total_steps)

            # print("Created the optimizer, scheduler and loss function objects in {} seconds".format(time.time() - st))
            print("--- Took %s seconds to create the model, optimizer, scheduler and loss function objects ---" % (time.time() - createOSL))

            print('##################################################################################')
            print('Begin training...')
            print('##################################################################################')
            train_start = time.time()
            # saved_models = train(model, optimizer, scheduler, train_dataloader, dev_dataloader, exp_args, run, eachSeed)
            print("--- Took %s seconds to train and evaluate the model ---" % (time.time() - train_start))

            # Use the saved models to evaluate on the test set
            # print( saved_models )

            print('##################################################################################')
            print('Begin test...')
            print('##################################################################################')
            # checkpoint = torch.load(saved_models[-1], map_location='cuda:0')
            checkpoint = torch.load('/mnt/nas2/results/Results/systematicReview/distant_pico/FS/participant/transformercrf/0_7.pth', map_location='cuda:0')
            model.load_state_dict( checkpoint )
            model = torch.nn.DataParallel(model, device_ids=[0])

            print('Applying the best model on test set (EBM-NLP)...')
            test1_cr, all_pred_flat, all_GT_flat, cm1, test1_words, class_rep_temp = evaluate(model, optimizer, scheduler, test1_dataloader, exp_args)
            print(test1_cr)
            print(cm1)

            print('Applying the best model on test set (Hilfiker et al.)...')
            test2_cr, all_pred_flat, all_GT_flat, cm2, test2_words, class_rep_temp = evaluate(model, optimizer, scheduler, test2_dataloader, exp_args)
            print(test2_cr)
            print(cm2)