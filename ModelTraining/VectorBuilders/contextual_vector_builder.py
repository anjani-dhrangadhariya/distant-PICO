##################################################################################
# Imports
##################################################################################
# staple imports
import warnings

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

