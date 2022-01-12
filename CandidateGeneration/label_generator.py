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



################################################################################
# Instantiate ElasticSearch
################################################################################
es = Elasticsearch( [{u'host': u'127.0.0.1', u'port': b'9200'}] )

################################################################################
# Get all the documents from the index
################################################################################
# Scan all of the CTO index (This index stores all the documents downloaded from clinicaltrials.org)
results_gen = helpers.scan(
    es,
    query={"query": {"match_all": {}}},
    index='ctofull-index',
    size=1000,
    scroll="60m",
)

match_scores = []
intervention_types = []

res = es.search(index="ctofull2021-index", body={"query": {"match_all": {}}}, size=10)
print('Total number of records retrieved: ', res['hits']['total']['value'])

# theFile ='/mnt/nas2/data/systematicReview/clinical_trials_gov/distant_pico_pre/secondary_outcomes.txt'
# theFile ='/home/anjani/distant-PICO/CandidateGeneration/ResultInspection/label_overlap_inspection.txt'
theFile = '/home/anjani/distant-PICO/CandidateGeneration/ResultInspection/resolve_annot_corpus.tsv'
aggregated_file = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/PICOS_data_preprocessed/aggregated_1_1.txt'
merged_file = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/PICOS_data_preprocessed/merged_1_1.txt'

# with open(aggregated_file, 'a+') as awf , open(merged_file, 'a+') as mwf:
with open(theFile, 'a+') as wf:

    all_tokens = []
    all_l1l2 = []

    # Iterate through all of the fetched CTO index documents
    for n, hit in enumerate( res['hits']['hits'] ): # Only a part search results from the CTO
    # for hit in results_gen: # Entire CTO

        hit_tokens = []
        hit_l1l2_labels = []
        sentence_mapper = []

        try:

            fullstudy = hit['_source']['FullStudiesResponse']['FullStudies'][0]['Study']
            NCT_id = hit['_source']['FullStudiesResponse']['Expression']
            # Annotation aggregator for PICOS for the NCT_id
            global_aggregator = dict()
            p_aggregator = dict()
            ic_aggregator = dict()
            o_aggregator = dict()
            s_aggregator = dict()

            protocol_section = fullstudy['ProtocolSection']
            derieved_section = fullstudy['DerivedSection']

            #############################################################################
            # Retrieve the distant supervision sources of PICOS annotation (TODO: This is some form of ETL if multiple data sources were involved!!!)
            #############################################################################
            participants = fetchParticipantSources(protocol_section)
            expanded_condition = expandCondition(participants['p_condition'], fetch_pos=True, fetch_abb=True)

            intervention_comparator = fetchIntcompSources(protocol_section)
            expanded_intervention = expandIntervention(intervention_comparator, fetch_pos=True, fetch_abb=True)

            outcomes = fetchOutcomeSources(protocol_section)
            expanded_prim_outcomes = expandOutcomes(outcomes, fetch_pos = True, fetch_abb = True)

            study_type = fetchStdTypeSources(protocol_section) # TODO: Expand the study type sources

            # Write the sources to a files





            # dump to file
            # json.dump(globally_aggregated, awf)
            # awf.write('\n')
            # json.dump(globally_merged, mwf)
            # mwf.write('\n')

            # log_success = 'Wrote the weak annotations for ' + str(NCT_id)
            # logging.info(log_success)  

        except Exception as ex:
          
            template = "An exception of type {0} occurred. Arguments:{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print( NCT_id , ' : ', message )

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

            print(traceback.format_exc())

            # logging.info(NCT_id)
            # logging.info(message)
            # string2log = str(exc_type) + ' : ' + str(fname) + ' : ' + str(exc_tb.tb_lineno)
            # logging.info(string2log)
            # logging.info(traceback.format_exc())
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