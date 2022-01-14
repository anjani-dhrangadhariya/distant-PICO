import argparse
from pathlib import Path
from random import random

# pyTorch essentials
import torch
import os
import random
import numpy as np

def getSources(annotations, lab):

    sources = []
    for k,v in annotations.items():

        more_than_source = len( v )
        for k_i, v_i in v.items():
            if 'source' in k_i and more_than_source > 1:
                sources.append( v_i )

    return sources, [lab] * len(sources)

def getArguments():

    # List of arguments to set up experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('-entity', type = str, default = 'i')
    parser.add_argument('-o_labeler1', type = bool, default = True)
    parser.add_argument('-o_labeler2', type = bool, default = True) # embed_type = {contextual, semantic} 
    parser.add_argument('-int_example', type = int, default= 10)
    parser.add_argument('-float_example', type = float, default= 1e-8)

    args = parser.parse_args()

    return args

def seed_everything( seed ):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True