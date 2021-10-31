import numpy as np
import pandas as pd
import random
import os
from mlflow import log_metric, log_param, log_artifacts
import mlflow
import time

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
from torch import LongTensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from read_candidates import fetchAndTransformCandidates
from Utilities.choosers import *

def convertDf2Tensor(df):

    return torch.from_numpy( np.array( list( df ), dtype=np.int64 )).clone().detach()


if __name__ == "__main__":


    for eachSeed in [ 0, 1, 42 ]:

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
            print('Size of training set: ', len(annotations.index))

            # Convert all inputs, labels, and attentions into torch tensors, the required datatype: torch.int64
            train_input_ids = convertDf2Tensor(annotations['embeddings'])
            train_input_labels = convertDf2Tensor(annotations['label_pads'])
            train_attn_masks = convertDf2Tensor(annotations['attn_masks'])
            train_pos_tags = convertDf2Tensor(annotations['inputpos'])
            # train_pos_tags = torch.nn.functional.one_hot( torch.from_numpy( np.array( list(annotations['inputpos']), dtype=np.int64) ).clone().detach() )

            assert train_input_ids.dtype == train_input_labels.dtype == train_attn_masks.dtype == train_pos_tags.dtype

            # ----------------------------------------------------------------------------------------
            # Create dataloaders from the tensors
            # ----------------------------------------------------------------------------------------
            # Create the DataLoader for our training set.
            train_data = TensorDataset(train_input_ids, train_input_labels, train_attn_masks, train_pos_tags)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=None, batch_size=10, shuffle=False)

            ##################################################################################
            #Instantiating the BERT model
            ##################################################################################
            print("Building model...")
            st = time.time()
            model = choose_model(exp_args.embed, current_tokenizer, exp_args.model, exp_args)

            