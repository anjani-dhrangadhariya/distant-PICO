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


if __name__ == "__main__":


    # for eachSeed in [ 0, 1, 42 ]:
    for eachSeed in [ 0 ]:

        with mlflow.start_run():

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
            annotations, exp_args, current_tokenizer = fetchAndTransformCandidates()

            if exp_args.train_data == 'distant-cto': 
                fulldf = annotations

            # shuffle the dataset and divide into training and validation sets
            fulldf = fulldf.sample(frac=1).reset_index(drop=True)
            annotations, annotations_devdf_ = train_test_split(fulldf, test_size=0.2) 
            print('Size of training set: ', len(annotations.index))
            print('Size of development set: ', len(annotations_devdf_.index))


            # Log the parameters
            logParams(exp_args)

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
            dev_dataloader = DataLoader(dev_data, sampler=None, batch_size=6, shuffle=False)

            ##################################################################################
            #Instantiating the BERT model
            ##################################################################################
            print("Building model...")
            createOSL = time.time()
            model = choose_model(exp_args.embed, current_tokenizer, exp_args.model, exp_args)

            ##################################################################################
            # Tell pytorch to run data on this model on the GPU and parallelize it
            ##################################################################################

            if exp_args.parallel == 'true':
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model, device_ids = [0, 1])
                    print("Using", len(model.device_ids), " GPUs!")
                    print("Using", str(model.device_ids), " GPUs!")

            elif exp_args.parallel == 'false':
                model = nn.DataParallel(model, device_ids = [0])
            
            print('Number of devices used: ', len(model.device_ids) )

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
                                                        num_warmup_steps=0,
                                                        num_training_steps = total_steps)

            # print("Created the optimizer, scheduler and loss function objects in {} seconds".format(time.time() - st))
            print("--- Took %s seconds to create the model, optimizer, scheduler and loss function objects ---" % (time.time() - createOSL))

            print('##################################################################################')
            print('Begin training...')
            print('##################################################################################')
            train_start = time.time()
            train(model, optimizer, scheduler, train_dataloader, dev_dataloader, exp_args, eachSeed)
            print("--- Took %s seconds to train and evaluate the model ---" % (time.time() - train_start))