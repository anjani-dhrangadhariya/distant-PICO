#!/usr/bin/env python

def SourceTargetsDoc(a):
    '''The module generates distant annotations for PICO entities using a cominbation of distant supervision and dynamic programming'''
    return a**a

print( SourceTargetsDoc.__doc__ )

import collections
import datetime as dt
import difflib
from functools import reduce
import itertools
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from random import shuffle
import operator

import matplotlib
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Q, Search

matplotlib.use('agg')
import matplotlib.pyplot as plt
from pylab import *
from sklearn.model_selection import train_test_split
from snorkel.labeling.model import LabelModel

from CandGenUtilities.experiment_arguments import *
from CandGenUtilities.labeler_utilities import *
from CandGenUtilities.source_target_mapping import *
from LabelingFunctions.ontologyLF import *
from Ontologies.ontologyLoader import *
from sanity_checks import *

'''
                for m in matches:
                    if m.span() in tokenized_spans:
                        spans_for_annot.append( m.span() )
                        matched_term.append( m.group() )
'''

################################################################################
# Initialize and set seed
################################################################################
# Get the experiment arguments
args = getArguments()

# Initialize LabelModel with correct cardinality
label_model = LabelModel(cardinality=4, verbose=True)

# Get the mappings between sources and their relevant targets
mapping = generateMapping()

# Get the mappings between PICOS entity and their labels and vice versa
PICOS = generateLabels()
PICOS_reverse = generateAntiLabels(PICOS)
abstain_options = abstainOption()

seed = 0
seed_everything(seed)
print('The random seed is set to: ', seed)

################################################################################
# Initialize Labeling function sources
################################################################################

def rankSAB(umls_d):

    keys = [k for k in umls_d.keys()] 

    shuffle(keys)

    umls_d_new = dict()

    for k in keys:
        umls_d_new[k] = umls_d[k]

    return umls_d_new

def list2Nested(l, nested_length):
    return [l[i:i+nested_length] for i in range(0, len(l), nested_length)]

def partitionRankedSAB(umls_d):

    keys = list(umls_d.keys())

    partitioned_lfs = [ ]

    for i in range( 0, len(keys) ):

        if i == 0 or i == len(keys):
            if i == 0:
                partitioned_lfs.append( keys )
            if i ==len(keys):
                temp3 = list2Nested(keys, 1)
                partitioned_lfs.append( temp3 )
        else:
            temp1, temp2 = keys[:i] , keys[i:]
            temp3 = list2Nested( keys[:i], 1)
            temp3.append( keys[i:] )
            partitioned_lfs.append( temp3 )

    return partitioned_lfs

hit_tokens = []
hit_l1l2_labels = []
sentence_mapper = []

try:

    umls_db = '/mnt/nas2/data/systematicReview/UMLS/english_subset/umls_preprocessed/umls_pre.db'
    
    print('Retrieving UMLS ontology arm (Preprocessing applied)')
    umls_p  = loadUMLSdb(umls_db, 'P')    
    # umls_i = loadUMLSdb(umls_db, 'I')
    # umls_o = loadUMLSdb(umls_db, 'O')
 
    print('Retrieving non-UMLS Ontologies  (Preprocessing applied)')
    p_DO, p_DO_syn = loadOnt( '/mnt/nas2/data/systematicReview/Ontologies/participant/DOID.csv', delim = ',', term_index = 1, term_syn_index = 2  )
    p_ctd, p_ctd_syn = loadOnt( '/mnt/nas2/data/systematicReview/Ontologies/participant/CTD_diseases.tsv', delim = '\t', term_index = 0, term_syn_index = 7 )
    i_ctd, i_ctd_syn = loadOnt( '/mnt/nas2/data/systematicReview/Ontologies/intervention/CTD_chemicals.tsv', delim = '\t', term_index = 0, term_syn_index = 7 )
    i_chebi, i_chebi_syn = loadOnt('/mnt/nas2/data/systematicReview/Ontologies/intervention/CHEBI.csv', delim = ',', term_index = 1, term_syn_index = 2  )

    print('Retrieving hand-crafted dictionaries')
    p_genders = loadDict('/mnt/nas2/data/systematicReview/Ontologies/participant/gender_sexuality.txt')
    i_comparator = loadDict('/mnt/nas2/data/systematicReview/Ontologies/intervention/comparator_dict.txt')

    print('Retrieving distant supervision dictionaries')
    indir_ds = '/mnt/nas2/data/systematicReview/ds_cto_dict'
    ds_participant = loadDS(indir_ds, 'participant')
    ds_intervention = loadDS(indir_ds, 'intervention')
    ds_intervention_syn = loadDS(indir_ds, 'intervention_syn')
    ds_outcome = loadDS(indir_ds, 'outcome')

    print('Retrieving ReGeX patterns')
    p_sampsize = loadPattern( 'samplesize' )
    p_agerange = loadPattern( 'age1' )
    p_agemax = loadPattern( 'age2' )

    # TODO: Retrieve external models


    corpus = []
    corpus_labels = []

    train_dir = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/groundtruth/ebm_nlp'
    with open(f'{train_dir}/{args.entity}/sentences.txt', 'r') as rf:
        for eachStudy in rf:
            data = json.loads(eachStudy)
            
            for k,v in data.items():
                corpus.append( [x.strip() for x in v[0]] )
                corpus_labels.append( [x.strip() for x in v[1]] )

    
    df = pd.DataFrame( {'text': corpus, 'labels': corpus_labels} )
    X_train, X_validation, y_train, y_validation = train_test_split(df['text'], df['labels'], test_size=0.20)
    X_validation_flatten = [item for sublist in list(X_validation) for item in sublist]

    text = ' '.join(X_validation_flatten)
    assert len(re.split(' ', text)) == len(X_validation_flatten) == len( list(WhitespaceTokenizer().span_tokenize(text)) )
    spans = list(WhitespaceTokenizer().span_tokenize(text))

    # Randomly choose an ontology to map
    ontology_SAB = list(umls_p.keys())
    key = ontology_SAB[2]

    # Rank the ontology based on coverage on the validation set
    ranked_umls_p = rankSAB( umls_p )

    # Combine the ontologies into labeling functions
    partitioned_umls_p = partitionRankedSAB( ranked_umls_p ) # Once best UMLS combination is obtained, use them as individual LF arms

    # UMLS Ontology labeling
    ont_matches, ont_labels = OntologyLabelingFunction( text, X_validation_flatten, spans, umls_p[key], picos=None, expand_term=True )

    # non-UMLS Ontology labeling
    p_DO_ont_matches, p_DO_ont_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, p_DO, picos='P', expand_term=True )
    p_DO_syn_ont_matches, p_DO_syn_ont_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, p_DO_syn, picos='P', expand_term=True )

    p_ctd_matches, p_ctd_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, p_ctd, picos='P', expand_term=True )
    p_ctd_syn_matches, p_ctd_syn_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, p_ctd_syn, picos='P', expand_term=True )

    i_ctd_matches, i_ctd_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, i_ctd, picos='I', expand_term=True )
    i_ctd_syn_matches, i_ctd_syn_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, i_ctd_syn, picos='I', expand_term=True )

    i_chebi_matches, i_chebi_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, i_chebi, picos='I', expand_term=True )
    i_chebi_syn_matches, i_chebi_syn_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, i_chebi_syn, picos='I', expand_term=True )

    # Distant Supervision labeling - This could fit with Dictionary Labeling function
    p_DS_matches, p_DS_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, ds_participant, picos='P', expand_term=True )
    i_ds_matches, i_ds_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, ds_intervention, picos='I', expand_term=True )
    i_syn_ds_matches, i_syn_ds_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, ds_intervention_syn, picos='I', expand_term=True )
    o_ds_matches, o_ds_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, ds_outcome, picos='O', expand_term=True )

    # Dictionary Labeling Function
    gender_matches, gender_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, p_genders, picos='P', expand_term=True )
    comparator_matches, comparator_labels  = OntologyLabelingFunction( text, X_validation_flatten, spans, i_comparator, picos='I', expand_term=True  )
    
    # TODO: ReGeX Labeling Function
    samplesize_matches, samplesize_labels = OntologyLabelingFunction( text, X_validation_flatten, spans, [p_sampsize], picos='P', expand_term=False )
    agerange_matches, agerange_labels = OntologyLabelingFunction( text, X_validation_flatten, spans, [p_agerange], picos='P', expand_term=False )
    agemax_matches, agemax_labels = OntologyLabelingFunction( text, X_validation_flatten, spans, [p_agemax], picos='P', expand_term=False )

    print( len( samplesize_matches ) )
    print( len( agerange_matches ) )
    print( len( agemax_matches ) )

    # TODO: External Model Labeling function



except Exception as ex:
    
    template = "An exception of type {0} occurred. Arguments:{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print( message )

    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)

    print(traceback.format_exc())

    logging.info(message)
    string2log = str(exc_type) + ' : ' + str(fname) + ' : ' + str(exc_tb.tb_lineno)
    logging.info(string2log)
    logging.info(traceback.format_exc())
    
'''
    ########
    all_tokens.append( hit_tokens )
    all_l1l2.extend( hit_l1l2_labels )


# print( all_l1l2 )
###### Label Model function here
L = np.array(all_l1l2)
label_model.fit(L, seed=100, n_epochs=100)
Y_hat = label_model.predict_proba(L)

print( Y_hat )
'''
