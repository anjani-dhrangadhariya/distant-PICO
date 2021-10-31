##################################################################################
# Imports
##################################################################################
# staple imports
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import datetime
import datetime as dt
import glob
import json
import logging
import os
import pdb
import random

# Memory leak
import gc

# statistics
import statistics
import sys
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# sklearn
import sklearn
from sklearn import preprocessing
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, plot_confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# Torch modules
from torch import LongTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
# Visualization
from tqdm import tqdm

# Transformers 
from transformers import (AdamW, AutoModel, AutoModelForTokenClassification,
                          AutoTokenizer, BertConfig, BertModel, BertTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer, RobertaConfig,
                          RobertaModel, get_linear_schedule_with_warmup)

import warnings
warnings.filterwarnings('ignore')

def printMetrics(cr):
    
    return cr['macro avg']['f1-score'], cr['1']['f1-score'], cr['2']['f1-score'], cr['3']['f1-score'], cr['4']['f1-score']

# Train
def train(defModel, optimizer, scheduler, train_dataloader, exp_args, eachSeed):

    with torch.enable_grad():

        for epoch_i in range(0, exp_args.max_eps):
            # Accumulate loss over an epoch
            total_train_loss = 0

            # (coarse-grained) accumulate predictions and labels over the epoch
            train_epoch_logits_coarse_i = []
            train_epochs_labels_coarse_i = []

            # Training for all the batches in this epoch
            for step, batch in enumerate(train_dataloader):

                # Clear the gradients
                optimizer.zero_grad()

                b_input_ids = batch[0].to(f'cuda:{defModel.device_ids[0]}')
                b_labels = batch[1].to(f'cuda:{defModel.device_ids[0]}')
                b_masks = batch[2].to(f'cuda:{defModel.device_ids[0]}')
                b_pos = batch[3].to(f'cuda:{defModel.device_ids[0]}')

                b_loss, b_output, b_labels, b_mask = defModel(input_ids = b_input_ids, attention_mask=b_masks, labels=b_labels, input_pos=b_pos)

                total_train_loss += abs( torch.mean(b_loss) ) 

                abs( torch.mean(b_loss) ).backward()

                # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(defModel.parameters(), 1.0)

                #Optimization step
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                for i in range(0, b_labels.shape[0]):

                    selected_preds_coarse = torch.masked_select( b_output[i, ].to(f'cuda:{defModel.device_ids[0]}'), b_mask[i, ])
                    selected_labs_coarse = torch.masked_select(b_labels[i, ].to(f'cuda:{defModel.device_ids[0]}'), b_mask[i, ])

                    train_epoch_logits_coarse_i.extend( selected_preds_coarse.to("cpu").numpy() )
                    train_epochs_labels_coarse_i.extend( selected_labs_coarse.to("cpu").numpy() )


                if step % exp_args.print_every == 0:

                    cr = sklearn.metrics.classification_report(y_pred= train_epoch_logits_coarse_i, y_true= train_epochs_labels_coarse_i, labels= list(range(5)), output_dict=True)
                    f1, f1_1 , f1_2, f1_3, f1_4 = printMetrics(cr)
                    print('Training: Epoch {} with macro average F1: {}, F1 score (P): {}, F1 score (IC): {}, F1 score (O), F1 score (S): {}'.format(epoch_i, f1, f1_1 , f1_2, f1_3, f1_4))

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            train_cr = classification_report(y_pred= train_epoch_logits_coarse_i, y_true=train_epochs_labels_coarse_i, labels= list(range(5)), output_dict=True)             

            # Delete the collected logits and labels
            del train_epoch_logits_coarse_i, train_epochs_labels_coarse_i
            gc.collect()

            