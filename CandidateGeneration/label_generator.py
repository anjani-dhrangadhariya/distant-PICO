#!/usr/bin/env python

def SourceTargetsDoc(a):
    '''The module generates distant annotations for PICO entities using a cominbation of distant supervision and dynamic programming'''
    return a**a

print( SourceTargetsDoc.__doc__ )

import collections
import datetime as dt
import difflib
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

import matplotlib
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Q, Search

matplotlib.use('agg')
import matplotlib.pyplot as plt
from pylab import *

from snorkel.labeling.model import LabelModel

# Import DISANT-PICO modules
from AnnotationAggregation.label_aggregator import *
from AnnotationAggregation.label_resolver import *
from AnnotationAggregation.sourcelevel_merging import *
from CandGenUtilities.experiment_arguments import *
from CandGenUtilities.labeler_utilities import *
from CandGenUtilities.source_target_mapping import *
from sanity_checks import *
from SourceFetcher.int_sourcefetcher import *
from SourceFetcher.outcome_sourcefetcher import *
from SourceFetcher.parti_sourcefetcher import *
from SourceFetcher.stdtype_sourcefetcher import *
from SourceTargetAligner.labeling_operators import *
from SourceTargetExpander.expand_sources import *
from SourceTargetExpander.expand_targets import *
from TargetFetcher.all_targetsfetcher import *
from SourceTargetAligner.labeling import *
from Ontologies.ontologyLoader import *

################################################################################
# Initialize 
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

################################################################################
# Initialize Labeling function sources
################################################################################

try:

    umls_db = '/mnt/nas2/data/systematicReview/UMLS/english_subset/umls_preprocessed/umls.db'
    
    # Retrieve the UMLS arm of PICOS annotation
    print('Retrieving UMLS ontology arm (Preprocessing applied)')
    # umls_p  = loadUMLSdb(umls_db, 'P')    
    # umls_i = loadUMLSdb(umls_db, 'I')
    # umls_o = loadUMLSdb(umls_db, 'O')

    # print( umls_p.keys() )

    # Retrieve non-UMLS Ontologies 
    p_DO, p_DO_syn = loadDO()
    p_ctd, p_ctd_syn = loadCTDdisease()
    i_ctd, i_ctd_syn = loadCTDchem()
    i_chebi, i_chebi_syn = loadChEBI()

    # Load external dictionaries
    p_genders = loadGenders()


    # TODO Retrieve distant supervision sources

except Exception as ex:
    
    template = "An exception of type {0} occurred. Arguments:{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print( message )

    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)

    print(traceback.format_exc())

hit_tokens = []
hit_l1l2_labels = []
sentence_mapper = []

try:

    corpus = []

    train_dir = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/groundtruth/ebm_nlp'
    with open(f'{train_dir}/{args.entity}/sentences.txt', 'r') as rf:
        for eachStudy in rf:
            data = json.loads(eachStudy)
            
            for k,v in data.items():
                corpus.extend( v[0] )

    

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