import numpy as np
import pandas as pd
import random
import os
from mlflow import log_metric, log_param, log_artifacts

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

if __name__ == "__main__":


    for eachSeed in [ 0, 1, 42 ]:

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
        annotations, exp_args = fetchAndTransformCandidates()
        print('Size of training set: ', len(annotations.index))


        








