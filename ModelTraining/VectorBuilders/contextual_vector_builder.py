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
from memory_profiler import profile
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
        elif n_subwords == 0:
            pass
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

##################################################################################
# Generates attention masks
##################################################################################
def createAttnMask(input_ids):
    # Add attention masks
    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return np.asarray(attention_masks, dtype=np.uint8)


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

    speTok_labels = addSpecialtokens(truncated_labels, 0, 0)
    speTok_pos = addSpecialtokens(truncated_pos, 0, 0)

    # PAD the sequences to max length
    if 'bert' in pretrained_model.lower():
        input_ids = pad_sequences([ speTok_sentence ] , maxlen=max_length, value=tokenizer.pad_token_id, padding="post")
    elif 'gpt2' in pretrained_model.lower():
        input_ids = pad_sequences([ speTok_sentence ] , maxlen=max_length, value=tokenizer.unk_token_id, padding="post")    

    input_labels = pad_sequences([ speTok_labels ] , maxlen=max_length, value=0, padding="post")
    input_pos = pad_sequences([ speTok_pos ] , maxlen=max_length, value=0, padding="post")

    assert len( input_ids ) == len( input_labels ) == len( input_pos )

    # Get the attention masks
    attention_masks = createAttnMask( input_ids )

    assert len(input_ids.squeeze()) == len(input_labels.squeeze()) == len(attention_masks.squeeze()) == len(input_pos.squeeze()) == max_length

    return input_ids.squeeze(), input_labels.squeeze(), attention_masks.squeeze(), input_pos.squeeze()

def getContextualVectors( annotations_df, tokenizer, pretrained_model, MAX_LEN, pos_encoder = None ):

    # Tokenize and preserve labels
    tokenized = []
    for tokens, labels, pos, nctid in zip(list(annotations_df['tokens']), list(annotations_df['labels']), list(annotations_df['pos']), list(annotations_df['ids'])) :
        temp = transform(tokens, labels, pos, tokenizer, MAX_LEN, pretrained_model)
        tokenized.append( temp )

    tokens, labels, masks, poss = list(zip(*tokenized))

    # Delete the tokenizer and tokenized list to reduce RAM usage
    del tokenized
    gc.collect()

    return tokens, labels, masks, poss, tokenizer