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

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from VectorBuilders.contextual_vector_builder import *


def readRawCandidates( list_NCT, label_type=None ):

    nct_ids = []
    tokens = []
    labels = []
    pos = []

    with open(list_NCT, 'r', encoding='latin1') as NCT_ids_file:

        for i, eachLine in enumerate(NCT_ids_file):
            annot = json.loads(eachLine)
            id_ = annot['id']

            for target_key, target in annot.items():

                if 'id' not in target_key:
                    for sentence_key, sentence in target.items():

                        tokens.append( sentence['tokens'] )
                        labels.append( sentence['annotation'] )
                        nct_ids.append( id_ )

                        # Generate dummy POS items
                        pos_i = [0] * len( sentence['tokens'] )
                        pos.append( pos_i )

            if i == 2:
                break

    corpus_df = pd.DataFrame(
        {'ids': nct_ids,
        'tokens': tokens,
        'labels': labels,
        'pos': pos
        })

    df = corpus_df.sample(frac=1).reset_index(drop=True) # Shuffles the dataframe after creation
    
    # can delete this one (corpusDf)
    del corpus_df
    gc.collect() # mark if for garbage collection

    return df

    return None


def fetchAndTransformCandidates():

    list_NCT = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/PICOS_data_preprocessed/merged_1_0.txt'

    MAX_LEN = 100

    start_candidate_reading = time.time()
    raw_candidates = readRawCandidates( list_NCT, label_type=None )
    print("--- Took %s seconds to read the raw weakly annotated candidates ---" % (time.time() - start_candidate_reading))

    start_candidate_transformation = time.time()
    input_embeddings, input_labels, input_masks, input_pos = getContextualVectors( raw_candidates, 'bert', MAX_LEN )
    assert len( input_embeddings ) == len( raw_candidates )
    print("--- Took %s seconds to transform the raw weakly annotated candidates ---" % (time.time() - start_candidate_transformation))