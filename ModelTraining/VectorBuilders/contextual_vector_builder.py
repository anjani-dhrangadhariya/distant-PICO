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

def tokenize_and_preserve_labels(sentence, text_labels, pos, tokenizer):
    dummy_label = 100 # Could be any kind of labels that you can mask
    tokenized_sentence = []
    labels = []
    poss = []
    printIt = []

    for word, label, pos_i in zip(sentence, text_labels, pos):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.encode(word, add_special_tokens = False)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        if n_subwords == 1:
            labels.extend([label] * n_subwords)
            poss.extend( [pos_i] * n_subwords ) 
        else:
            labels.extend([label])
            labels.extend( [dummy_label] * (n_subwords-1) )
            poss.extend( [pos_i] * n_subwords ) 

    assert len(tokenized_sentence) == len(labels) == len(poss)

    return tokenized_sentence, labels, poss

##################################################################################
# The function truncates input sequences to max lengths
##################################################################################
def truncateSentence(sentence, trim_len):

    trimmedSentence = []
    if  len(sentence) > trim_len:
        trimmedSentence = sentence[:trim_len]
    else:
        trimmedSentence = sentence

    assert len(trimmedSentence) <= trim_len
    return trimmedSentence

##################################################################################
# The function adds special tokens to the truncated sequences
##################################################################################
def addSpecialtokens(eachText, start_token, end_token):
    insert_at_start = 0
    eachText[insert_at_start:insert_at_start] = [start_token]

    insert_at_end = len(eachText)
    eachText[insert_at_end:insert_at_end] = [end_token]

    assert eachText[0] == start_token
    assert eachText[-1] == end_token

    return eachText

def transform(sentence, text_labels, pos, tokenizer, max_length, pretrained_model):

    # Tokenize and preserve labels
    tokenized_sentence, labels, poss = tokenize_and_preserve_labels(sentence, text_labels, pos, tokenizer)

    # Truncate the sequences (sentence and label) to (max_length - 2)
    if max_length >= 510:
        truncated_sentence = truncateSentence(tokenized_sentence, (max_length - 2))
        truncated_labels = truncateSentence(labels, (max_length - 2))
        truncated_pos = truncateSentence(poss, (max_length - 2))
        assert len(truncated_sentence) == len(truncated_labels) == len(truncated_pos)
    else:
        truncated_sentence = tokenized_sentence
        truncated_labels = labels
        truncated_pos = poss
        assert len(truncated_sentence) == len(truncated_labels) == len(truncated_pos)

    # Add special tokens CLS and SEP for the BERT tokenizer (identical for SCIBERT)
    if 'bert' in pretrained_model.lower():
        speTok_sentence = addSpecialtokens(truncated_sentence, tokenizer.cls_token_id, tokenizer.sep_token_id)
    elif 'gpt2' in pretrained_model.lower():
        speTok_sentence = addSpecialtokens(truncated_sentence, tokenizer.bos_token_id, tokenizer.eos_token_id)


    


    print( speTok_sentence )

    return None


def getContextualVectors( annotations_df, vector_type, MAX_LEN, pos_encoder = None ):

    tokenizer = choose_tokenizer_type( vector_type )

    # Tokenize and preserve labels
    tokenized = []
    for tokens, labels, pos in zip(list(annotations_df['tokens']), list(annotations_df['labels']), list(annotations_df['pos'])) :
        temp = transform(tokens, labels, pos, tokenizer, MAX_LEN, vector_type)
        tokenized.append( temp )

    return None