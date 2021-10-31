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


# Train
def train(defModel, optimizer, scheduler, train_dataloader, exp_args, eachSeed):

    with torch.enable_grad():

        for epoch_i in range(0, exp_args.max_eps):
            # Accumulate loss over an epoch
            total_train_loss = 0

            # Training for all the batches in this epoch
            for step, batch in enumerate(train_dataloader):

                # Clear the gradients
                optimizer.zero_grad()

                b_input_ids = batch[0].to(f'cuda:{defModel.device_ids[0]}')
                b_labels = batch[1].to(f'cuda:{defModel.device_ids[0]}')
                b_masks = batch[2].to(f'cuda:{defModel.device_ids[0]}')
                b_pos = batch[3].to(f'cuda:{defModel.device_ids[0]}')

                b_loss, b_output, b_labels, b_mask = defModel(input_ids = b_input_ids, attention_mask=b_masks, labels=b_labels, input_pos=b_pos)                




