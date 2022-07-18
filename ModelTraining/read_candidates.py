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

from Utilities.experiment_arguments import *
from VectorBuilders.contextual_vector_builder import *


# @profile
def readRawCandidates( raw_cand_file, label_type=None ):

    corpus_df = pd.read_csv(raw_cand_file, sep='\t', header=0)

    # create dummy pos tags TODO: can remove it
    pos_tags = corpus_df['pos']
    dummy_pos_tags = []
    for i in pos_tags:
        j = ast.literal_eval( i )
        dummy = [0] * len(j)
        dummy_pos_tags.append( str(dummy) )
    
    corpus_df['pos'] = dummy_pos_tags

    return corpus_df

def readManuallyAnnoted( input_file_path, entity, label_type=None ):

    nct_ids = []
    tokens = []
    labels = []
    labels_fine = []
    pos = []

    with open(input_file_path, 'r', encoding='latin1') as NCT_ids_file:

        for i, eachLine in enumerate(NCT_ids_file):
            annot = json.loads(eachLine)

            for doc_key, document_annotations in annot.items():

                nct_ids.append(doc_key)
                tokens.append(document_annotations['tokens'] )
                entity_mod = entity + str('s')
                if entity_mod in document_annotations:
                    labels.append(document_annotations[entity_mod])
                else:
                    dummy_labels = ['0'] * len(document_annotations['tokens'])
                    labels.append(dummy_labels)
                
                fine_entity = entity_mod + str('_fine')
                if fine_entity in document_annotations:
                    fine_labels = document_annotations[fine_entity]

                    for counter, f_l in enumerate( fine_labels ):
                        if f_l == '0':
                            fine_labels[counter] = '0'
                        if f_l != '0' and f_l != '0' and f_l != '1':
                            fine_labels[counter] = '1'


                    labels_fine.append(fine_labels)
                else:
                    dummy_labels = ['0'] * len(document_annotations['tokens'])
                    labels_fine.append(dummy_labels)

                dummy_pos = ['0'] * len(document_annotations['pos'])
                pos.append( dummy_pos )


    corpus_df = pd.DataFrame(
        {'ids': nct_ids,
        'tokens': tokens,
        'labels': labels_fine, #XXX : exchanged labels and labels_fine with attriute : value
        'labels_fine': labels,
        'pos': pos
        })

    return corpus_df


def fetchAndTransformCandidates():

    args = getArguments() # get all the experimental arguments

    start_candidate_reading = time.time()
    raw_candidates = readRawCandidates( args.rawcand_file, label_type=None )
    print("--- Took %s seconds to read the raw weakly annotated candidates ---" % (time.time() - start_candidate_reading))

    # # Retrieve EBM-PICO dataset here
    start_manual_reading = time.time()
    ebm_gold = readManuallyAnnoted( args.ebm_gold, entity=args.entity, label_type=None)
    hilfiker = readManuallyAnnoted( args.hilfiker, entity=args.entity, label_type=None)
    print("--- Took %s seconds to read the manually annotated datasets ---" % (time.time() - start_manual_reading))

    start_candidate_transformation = time.time()
    tokenizer, model = choose_tokenizer_type( args.embed )
    input_embeddings, input_labels, input_masks, input_pos, tokenizer = getContextualVectors( raw_candidates, tokenizer, args.embed, args.max_len )
    assert len( input_embeddings ) == len( raw_candidates )
    print("--- Took %s seconds to transform the raw weakly annotated candidates ---" % (time.time() - start_candidate_transformation))

    start_manual_transformation = time.time()
    ebm_gold_embeddings, ebm_gold_labels, ebm_gold_masks, ebm_gold_pos, tokenizer = getContextualVectors( ebm_gold, tokenizer, args.embed, args.max_len )
    hilfiker_embeddings, hilfiker_labels, hilfiker_masks, hilfiker_pos, tokenizer = getContextualVectors( hilfiker, tokenizer, args.embed, args.max_len )
    print("--- Took %s seconds to transform the manually annotated datasets ---" % (time.time() - start_manual_transformation))

    candidates_df = raw_candidates.assign(embeddings = pd.Series(input_embeddings).values, label_pads = pd.Series(input_labels).values, attn_masks = pd.Series(input_masks).values, inputpos = pd.Series(input_pos).values) # assign the padded embeddings to the dataframe
    ebm_gold_df = ebm_gold.assign(embeddings = pd.Series(ebm_gold_embeddings).values, label_pads = pd.Series(ebm_gold_labels).values, attn_masks = pd.Series(ebm_gold_masks).values, inputpos = pd.Series(ebm_gold_pos).values)
    hilfiker_df = hilfiker.assign(embeddings = pd.Series(hilfiker_embeddings).values, label_pads = pd.Series(hilfiker_labels).values, attn_masks = pd.Series(hilfiker_masks).values, inputpos = pd.Series(hilfiker_pos).values)

    return candidates_df, ebm_gold_df, hilfiker_df, args, tokenizer, model