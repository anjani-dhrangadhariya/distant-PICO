
##################################################################################
# Imports
##################################################################################
# staple imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import glob
import numpy as np
import pandas as pd
import time
import datetime
import argparse
import pdb
import json
import random

# numpy essentials
from numpy import asarray
import numpy as np

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
from torch import LongTensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# keras essentials
from keras.preprocessing.sequence import pad_sequences

# sklearn
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix

# sklearn crfsuite
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# pyTorch CRF
from torchcrf import CRF

# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig, BertForTokenClassification
from transformers import AdamW, BertConfig 
from transformers import get_linear_schedule_with_warmup

def get_packed_padded_output(b_sequence_output, b_input_ids, b_input_mask, tokenizer):
    
    # Sequence lengths to sort and to let LSTM know what are <PAD> tokens
    sequence_lengths = b_input_mask.sum(dim = 1) # 0 = PAD, 1 = other

    # Sort masked padded embedding outputs according to sequence length
    seq_lengths, perm_idx = sequence_lengths.sort(0, descending=True)
    b_masked_padded_sorted_output_ = b_sequence_output[perm_idx]

    # Pack the sorted masked padded output for input to the LSTM layer
    packed_input = pack_padded_sequence(b_masked_padded_sorted_output_, seq_lengths.cpu(), batch_first=True)
    # packed_input.to(f'cuda:{model.device_ids[0]}')

    return packed_input, perm_idx, seq_lengths


def get_packed_padded_output_dataparallel(b_sequence_output, b_input_ids, b_input_mask, tokenizer):
    
    # Sequence lengths to sort and to let LSTM know what are <PAD> tokens
    sequence_lengths = b_input_mask.sum(dim = 0) # 0 = PAD, 1 = other

    # Sort masked padded embedding outputs according to sequence length
    seq_lengths, perm_idx = sequence_lengths.sort(0, descending=True)
    b_masked_padded_sorted_output_ = b_sequence_output[perm_idx]

    # Get full length of the sequence
    total_length = b_masked_padded_sorted_output_.size(1)

    # Pack the sorted masked padded output for input to the LSTM layer
    packed_input = pack_padded_sequence(b_masked_padded_sorted_output_, seq_lengths.cpu(), batch_first=True)
    # packed_input.to(f'cuda:{model.device_ids[0]}')

    return packed_input, perm_idx, seq_lengths, total_length