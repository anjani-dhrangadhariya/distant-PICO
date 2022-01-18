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
from itertools import chain

import matplotlib
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Q, Search
from nltk.tokenize import WhitespaceTokenizer

matplotlib.use('agg')
import matplotlib.pyplot as plt
from pylab import *
from snorkel.labeling.model import LabelModel

from CandGenUtilities.experiment_arguments import *
from CandGenUtilities.labeler_utilities import *
from CandGenUtilities.source_target_mapping import *
from LabelingFunctions.ontologyLF import *
from Ontologies.ontologyLoader import *
from LabelingFunctions.LFutils import spansToLabels
from LabelingFunctions.externalmodelLF import ExternalModelLabelingFunction
from LabelingFunctions.heuristicLF import heurPattern_pa, posPattern_i
from load_data import loadEBMPICO

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
    #umls_p  = loadUMLSdb(umls_db, 'P')    
    # umls_i = loadUMLSdb(umls_db, 'I')
    #umls_o = loadUMLSdb(umls_db, 'O')
 
    print('Retrieving non-UMLS Ontologies  (Preprocessing applied)')
    p_DO, p_DO_syn = loadOnt( '/mnt/nas2/data/systematicReview/Ontologies/participant/DOID.csv', delim = ',', term_index = 1, term_syn_index = 2  )
    p_ctd, p_ctd_syn = loadOnt( '/mnt/nas2/data/systematicReview/Ontologies/participant/CTD_diseases.tsv', delim = '\t', term_index = 0, term_syn_index = 7 )
    i_ctd, i_ctd_syn = loadOnt( '/mnt/nas2/data/systematicReview/Ontologies/intervention/CTD_chemicals.tsv', delim = '\t', term_index = 0, term_syn_index = 7 )
    i_chebi, i_chebi_syn = loadOnt('/mnt/nas2/data/systematicReview/Ontologies/intervention/CHEBI.csv', delim = ',', term_index = 1, term_syn_index = 2  )
    o_oae, o_oae_syn = loadOnt('/mnt/nas2/data/systematicReview/Ontologies/outcome/OAE.csv', delim=',', term_index=1, term_syn_index=2 )

    print('Retrieving hand-crafted dictionaries')
    p_genders = loadDict('/mnt/nas2/data/systematicReview/Ontologies/participant/gender_sexuality.txt')
    i_comparator = loadDict('/mnt/nas2/data/systematicReview/Ontologies/intervention/comparator_dict.txt')

    print('Retrieving distant supervision dictionaries')
    indir_ds = '/mnt/nas2/data/systematicReview/ds_cto_dict'
    ds_participant = loadDS(indir_ds, 'participant')
    ds_intervention = loadDS(indir_ds, 'intervention')
    ds_intervention_syn = loadDS(indir_ds, 'intervention_syn')
    ds_outcome = loadDS(indir_ds, 'outcome')

    print('Retrieving abbreviations dictionaries')
    p_abb = loadAbbreviations('/mnt/nas2/data/systematicReview/Ontologies/participant/diseaseabbreviations.tsv')

    print('Retrieving ReGeX patterns')
    p_sampsize = loadPattern( 'samplesize' )
    p_agerange = loadPattern( 'age1' )
    p_agemax = loadPattern( 'age2' )

    # TODO: Retrieve external models


    # Load validation data
    ebm_nlp = '/mnt/nas2/data/systematicReview/PICO_datasets/EBM_parsed'
    train, validation = loadEBMPICO( ebm_nlp )
    
    validation_token_flatten = [item for sublist in list(validation['tokens']) for item in sublist]
    validation_pos_flatten = [item for sublist in list(validation['pos']) for item in sublist]

    validation_p_labels_flatten = [item for sublist in list(validation['p']) for item in sublist]
    validation_i_labels_flatten = [item for sublist in list(validation['i']) for item in sublist]
    validation_o_labels_flatten = [item for sublist in list(validation['o']) for item in sublist]

    text = ' '.join(validation_token_flatten)
    assert len(re.split(' ', text)) == len(validation_token_flatten) == len( list(WhitespaceTokenizer().span_tokenize(text)) )
    spans = list(WhitespaceTokenizer().span_tokenize(text))
    start_spans = [y[0] for y in spans]
    end_spans = [y[1] for y in spans]

    start_spans = dict()
    for i, y in  enumerate(spans):
        ss = y[0]
        es = y[1]

        for value in range( ss, es+1 ):
            start_spans[value] = i



    # Randomly choose an ontology to map
    # ontology_SAB = list(umls_i.keys())
    # key = ontology_SAB[4]

    # Rank the ontology based on coverage on the validation set
    #ranked_umls_p = rankSAB( umls_p )
    # ranked_umls_i = rankSAB( umls_i )
    #ranked_umls_o = rankSAB( umls_o )


    # Combine the ontologies into labeling functions
    # partitioned_umls_i = partitionRankedSAB( ranked_umls_i ) # Once best UMLS combination is obtained, use them as individual LF arms

    #########################################################################################
    # Level 1 - UMLS LF's
    #########################################################################################
    # UMLS Ontology labeling
    #ont_p_matches, ont_p_labels = OntologyLabelingFunction( text, validation_token_flatten, spans, umls_p[key], picos=None, expand_term=True, fuzzy_match=False )
    # ont_i_matches, ont_i_labels = OntologyLabelingFunction( text, validation_token_flatten, spans, umls_i[key], picos=None, expand_term=True, fuzzy_match=False )
    #ont_o_matches, ont_o_labels = OntologyLabelingFunction( text, validation_token_flatten, spans, umls_o[key], picos=None, expand_term=True, fuzzy_match=False )
    
    
    #########################################################################################
    # Level 2 - Non-UMLS LF's
    #########################################################################################
    # non-UMLS Ontology labeling
    p_DO_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, p_DO, picos='P', expand_term=True, fuzzy_match = False )
    p_DO_syn_labels = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, p_DO_syn, picos='P', expand_term=True, fuzzy_match = False )
    '''
    # p_ctd_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, p_ctd, picos='P', expand_term=True , fuzzy_match = False)
    # p_ctd_syn_labels = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, p_ctd_syn, picos='P', expand_term=True, fuzzy_match = False )


    i_ctd_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, i_ctd, picos='I', expand_term=True, fuzzy_match = False )
    # i_ctd_syn_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, i_ctd_syn, picos='I', expand_term=True, fuzzy_match = False )

    i_chebi_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, i_chebi, picos='I', expand_term=True , fuzzy_match = False)
    i_chebi_syn_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, i_chebi_syn, picos='I', expand_term=True, fuzzy_match = False )

    o_oae_labels = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, o_oae, picos='O', expand_term=True , fuzzy_match = False)
    o_oae_syn_labels = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, o_oae_syn, picos='O', expand_term=True, fuzzy_match = False )


    #########################################################################################
    # Level 3 - Distant Supervision and hand-crafted dictionary LF's
    #########################################################################################
    # Distant Supervision labeling - This could fit with Dictionary Labeling function
    p_DS_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, ds_participant, picos='P', expand_term=True, fuzzy_match = False )
    i_ds_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, ds_intervention, picos='I', expand_term=True, fuzzy_match = False )
    i_syn_ds_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, ds_intervention_syn, picos='I', expand_term=True, fuzzy_match = False )
    o_ds_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, ds_outcome, picos='O', expand_term=True, fuzzy_match = False )

    # Dictionary Labeling Function
    gender_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, p_genders, picos='P', expand_term=True, fuzzy_match = False )
    comparator_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, i_comparator, picos='I', expand_term=True, fuzzy_match = False  )

    # Abbreviation dictionary Labeling function
    p_abb_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, p_abb, picos='P', expand_term=False, fuzzy_match = False )


    #########################################################################################
    # Level 4 - Rule based LF's (ReGeX, Heuristics, Ontology based fuzzy bigram match)
    #########################################################################################
    # ReGeX Labeling Function
    samplesize_labels = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, [p_sampsize], picos='P', expand_term=False, fuzzy_match = False )
    agerange_labels = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, [p_agerange], picos='P', expand_term=False, fuzzy_match = False )
    agemax_labels = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, [p_agemax], picos='P', expand_term=False, fuzzy_match = False )

    # Heutistic Labeling Function
    #i_posregMatches, i_posregSpans, i_posregLabels = posPattern_i( text, validation_token_flatten, validation_pos_flatten, spans, picos='I' )
    #pa_regex_heur_matches, pa_regex_heur_labels = heurPattern_pa( text, validation_token_flatten, validation_pos_flatten, spans, picos='I' )

    # Fuzzy ontology LFs
    p_DO_fz_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, p_DO, picos='P', expand_term=True, fuzzy_match = True )
    p_DO_syn_fz_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, p_DO_syn, picos='P', expand_term=True, fuzzy_match = True )

    p_ctd_fz_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, p_ctd, picos='P', expand_term=True , fuzzy_match = True)
    p_ctd_syn_fz_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, p_ctd_syn, picos='P', expand_term=True, fuzzy_match = True )

    i_ctd_fz_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, i_ctd, picos='I', expand_term=True, fuzzy_match = True )
    i_ctd_syn_fz_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, i_ctd_syn, picos='I', expand_term=True, fuzzy_match = True )

    i_chebi_fz_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, i_chebi, picos='I', expand_term=True , fuzzy_match = True )
    i_chebi_syn_fz_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, i_chebi_syn, picos='I', expand_term=True, fuzzy_match = True )

    o_oae_fz_labels = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, o_oae, picos='O', expand_term=True , fuzzy_match = True )
    o_oae_syn_fz_labels = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, o_oae_syn, picos='O', expand_term=True, fuzzy_match = True )

    p_DS_fz_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, ds_participant, picos='P', expand_term=True, fuzzy_match = True )
    i_ds_fz_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, ds_intervention, picos='I', expand_term=True, fuzzy_match = True )
    i_syn_ds_fz_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, ds_intervention_syn, picos='I', expand_term=True, fuzzy_match = True )
    o_ds_fz_labels  = OntologyLabelingFunction( text, validation_token_flatten, spans, start_spans, ds_outcome, picos='O', expand_term=True, fuzzy_match = True )


    #########################################################################################
    # TODO  Level 5 - External Model Labeling function
    #########################################################################################
    ExternalModelLabelingFunction()
    '''
    



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
