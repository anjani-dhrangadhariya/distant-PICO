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
from memory_profiler import profile
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from VectorBuilders.contextual_vector_builder import *
from Utilities.experiment_arguments import *

# @profile
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

                        if set(sentence['tokens'])!={0}:
                            tokens.append( sentence['tokens'] )
                            labels.append( sentence['annotation'] )
                            nct_ids.append( id_ )

                            # TODO: Generate dummy POS items
                            pos_i = [0] * len( sentence['tokens'] )
                            pos.append( pos_i )
                        else:
                            print('All the labels are nil')

            if i == 6:
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

def readManuallyAnnoted( input_file_path, label_type=None ):

    nct_ids = []
    tokens = []
    labels = []
    pos = []

    with open(input_file_path, 'r', encoding='latin1') as NCT_ids_file:

        for i, eachLine in enumerate(NCT_ids_file):
            annot = json.loads(eachLine)

            for doc_key, document_annotations in annot.items():

                nct_ids.append(doc_key)
                tokens.append(document_annotations[0])
                labels.append(document_annotations[1])
                # TODO: Generate dummy POS items
                pos_i = [0] * len( document_annotations[0] )
                pos.append( pos_i )

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


def fetchAndTransformCandidates():

    args = getArguments() # get all the experimental arguments

    start_candidate_reading = time.time()
    raw_candidates = readRawCandidates( args.rawcand_file, label_type=None )
    print("--- Took %s seconds to read the raw weakly annotated candidates ---" % (time.time() - start_candidate_reading))

    # Retrieve EBM-PICO dataset here
    start_manual_reading = time.time()
    ebm_nlp = readManuallyAnnoted( args.ebm_nlp, label_type=None )
    ebm_gold = readManuallyAnnoted( args.ebm_gold, label_type=None )
    hilfiker = readManuallyAnnoted( args.hilfiker, label_type=None )
    print("--- Took %s seconds to read the manually annotated datasets ---" % (time.time() - start_manual_reading))

    start_candidate_transformation = time.time()
    tokenizer, model = choose_tokenizer_type( args.embed )
    input_embeddings, input_labels, input_masks, input_pos, tokenizer = getContextualVectors( raw_candidates, tokenizer, args.embed, args.max_len )
    assert len( input_embeddings ) == len( raw_candidates )
    print("--- Took %s seconds to transform the raw weakly annotated candidates ---" % (time.time() - start_candidate_transformation))

    start_manual_transformation = time.time()
    ebm_nlp_embeddings, ebm_nlp_labels, ebm_nlp_masks, ebm_nlp_pos, tokenizer = getContextualVectors( ebm_nlp, tokenizer, args.embed, args.max_len )
    ebm_gold_embeddings, ebm_gold_labels, ebm_gold_masks, ebm_gold_pos, tokenizer = getContextualVectors( ebm_gold, tokenizer, args.embed, args.max_len )
    hilfiker_embeddings, hilfiker_labels, hilfiker_masks, hilfiker_pos, tokenizer = getContextualVectors( hilfiker, tokenizer, args.embed, args.max_len )
    print("--- Took %s seconds to transform the manually annotated datasets ---" % (time.time() - start_manual_transformation))

    candidates_df = raw_candidates.assign(embeddings = pd.Series(input_embeddings).values, label_pads = pd.Series(input_labels).values, attn_masks = pd.Series(input_masks).values, inputpos = pd.Series(input_pos).values) # assign the padded embeddings to the dataframe
    ebm_nlp_df = ebm_nlp.assign(embeddings = pd.Series(ebm_nlp_embeddings).values, label_pads = pd.Series(ebm_nlp_labels).values, attn_masks = pd.Series(ebm_nlp_masks).values, inputpos = pd.Series(ebm_nlp_pos).values)
    ebm_gold_df = ebm_gold.assign(embeddings = pd.Series(ebm_gold_embeddings).values, label_pads = pd.Series(ebm_gold_labels).values, attn_masks = pd.Series(ebm_gold_masks).values, inputpos = pd.Series(ebm_gold_pos).values)
    hilfiker_df = hilfiker.assign(embeddings = pd.Series(hilfiker_embeddings).values, label_pads = pd.Series(hilfiker_labels).values, attn_masks = pd.Series(hilfiker_masks).values, inputpos = pd.Series(hilfiker_pos).values)

    del input_embeddings, input_labels, input_masks, input_pos     # Delete the large variables

    return candidates_df, ebm_nlp_df, ebm_gold_df, hilfiker_df, args, tokenizer, model