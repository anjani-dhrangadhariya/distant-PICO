##################################################################################
# Imports
##################################################################################
# staple imports
import warnings

from Utilities.choosers import choose_tokenizer_type

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import datetime
import datetime as dt
import gc
import glob
import json
import logging
import os
import pdb
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# visualization
import seaborn as sn
# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# keras essentials
from keras.preprocessing.sequence import pad_sequences
# numpy essentials
from numpy import asarray
from torch import LongTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (AdamW, AutoTokenizer, BertConfig, BertModel,
                          BertTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer,
                          RobertaConfig, RobertaModel,
                          get_linear_schedule_with_warmup)
from VectorBuilders.contextual_vector_builder import *


def transform():

    return None


def getContextualVectors( annotations_df, vector_type, MAX_LEN, pos_encoder = None ):

    tokenizer = choose_tokenizer_type( vector_type )

    # Tokenize and preserve labels
    tokenized = []
    for tokens, labels, pos in zip(list(annotations_df['tokens']), list(annotations_df['labels']), list(annotations_df['pos'])) :
        temp = transform(tokens, labels, pos, tokenizer, MAX_LEN, vector_type)
        tokenized.append( temp )

    tokens, labels, masks, poss = list(zip(*tokenized))

    # Delete the tokenizer and tokenized list to reduce RAM usage
    del tokenizer, tokenized
    gc.collect()

    return tokens, labels, masks, poss # Returns input IDs and labels together