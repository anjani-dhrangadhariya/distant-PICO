#!/usr/bin/env python

def SourceTargetsDoc(a):
    '''The module generates distant annotations for PICO entities using a cominbation of distant supervision and dynamic programming'''
    return a**a

print( SourceTargetsDoc.__doc__ )

__author__ = "Anjani Dhrangadhariya"
__maintainer__ = "Anjani Dhrangadhariya"
__email__ = "anjani.dhrangadhariya@hevs.ch"
__status__ = "Prototype"

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

from AnnotationAggregation.label_aggregator import *
from AnnotationAggregation.label_resolver import *
from sanity_checks import *
from SourceFetcher.int_sourcefetcher import *
from SourceFetcher.outcome_sourcefetcher import *

# Import DISANT-PICO modules
from SourceFetcher.parti_sourcefetcher import *
from SourceFetcher.stdtype_sourcefetcher import *
from SourceTargetAligner.labeling_operators import *
from SourceTargetAligner.source_target_mapping import *
from SourceTargetExpander.expand_sources import *
from SourceTargetExpander.expand_targets import *
from TargetFetcher.all_targetsfetcher import *

################################################################################
# Set the logger here
################################################################################
# LOG_FILE = os.getcwd() + "/logs"
# if not os.path.exists(LOG_FILE):
#     os.makedirs(LOG_FILE)

# LOG_FILE = LOG_FILE + "/" + dt.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d %H_%M_%S') + ".log"
# logFormatter = logging.Formatter("%(levelname)s %(asctime)s %(processName)s %(message)s")
# fileHandler = logging.FileHandler("{0}".format(LOG_FILE))
# rootLogger = logging.getLogger()
# rootLogger.addHandler(fileHandler)
# rootLogger.setLevel(logging.INFO)

################################################################################
def getSources(annotations, lab):

    sources = []
    for k,v in annotations.items():

        more_than_source = len( v )
        for k_i, v_i in v.items():
            if 'source' in k_i and more_than_source > 1:
                sources.append( v_i )

    return sources, [lab] * len(sources)

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

    # Iterate through all of the fetched CTO index documents
    for n, hit in enumerate( res['hits']['hits'] ): # Only a part search results from the CTO
    # for hit in results_gen: # Entire CTO

        try:

            fullstudy = hit['_source']['FullStudiesResponse']['FullStudies'][0]['Study']
            NCT_id = hit['_source']['FullStudiesResponse']['Expression']
            # print('################################## ', NCT_id , ' #########################################')
            # Annotation aggregator for PICOS for the NCT_id
            global_aggregator = dict()
            p_aggregator = dict()
            ic_aggregator = dict()
            o_aggregator = dict()
            s_aggregator = dict()

            protocol_section = fullstudy['ProtocolSection']
            derieved_section = fullstudy['DerivedSection']

            # Retrieve the sources of PICOS annotation (XXX: This is some form of ETL if multiple data sources were involved!!!)
            participants = fetchParticipantSources(protocol_section)
            intervention_comparator = fetchIntcompSources(protocol_section)
            outcomes = fetchOutcomeSources(protocol_section)
            study_type = fetchStdTypeSources(protocol_section)

            sources = {**participants, **intervention_comparator, **outcomes, **study_type}

            # Retrieve the targets of PICOS annotation
            targets = fetchTargets(protocol_section)

            # Expand the sources of PICOS annotation
            expanded_sources = expandSources(protocol_section, sources)

            # Expand the targets of PICOS annotation
            expanded_targets = expandTargets(protocol_section, targets)

            # Get the mappings between sources and their relevant targets
            mapping = generateMapping()

            # Get the mappings between PICOS entity and their labels and vice versa
            PICOS = generateLabels()
            PICOS_reverse = generateAntiLabels(PICOS)

            #################################################################
            # Direct matching begins
            # P = 1, I/C = 2, O = 3, S = 4
            # expanded_sources: All the sources from a NCTID; expanded_targets: All the targets from a NCTID
            #################################################################  

            sources_list = []
            source_labs = []
            annotation_collector = []

            for key, value in expanded_sources.items():

                if 'gender' in key:
                    candidate_targets = mapping[key]
                    gender_annotations = directAligner( value, expanded_targets, candidate_targets, PICOS['P'] )
                    if gender_annotations and len( getSecOrdKeys(gender_annotations) ) > 1:
                        annotation_collector.append( gender_annotations )
                    #     print( gender_annotations )


                if 'sample_size' in key:
                    candidate_targets = mapping[key]
                    sampsize_annotations = directAligner( [value], expanded_targets, candidate_targets, PICOS['P'] )   # direct aligner expects values as lists       
                    if sampsize_annotations and len( getSecOrdKeys(sampsize_annotations) ) > 1:
                        annotation_collector.append( sampsize_annotations )

                if 'age' in key:
                    candidate_targets = mapping[key]
                    stdage_annotations = directAligner( value['StdAge'], expanded_targets, candidate_targets, PICOS['P'] )
                    if stdage_annotations and len( getSecOrdKeys(stdage_annotations) ) > 1:
                        annotation_collector.append( stdage_annotations )
                    #     print( stdage_annotations )

                    if 'exactAge' in value:
                        exctage_annotattions = regexAligner( [value['exactAge']], expanded_targets, candidate_targets, PICOS['P'] ) # reGeX aligner expects values as lists   
                        if exctage_annotattions and len( getSecOrdKeys(exctage_annotattions) ) > 1:
                            annotation_collector.append( exctage_annotattions )                 

                if 'condition' in key:
                    candidate_targets = mapping[key]
                    condition_annotations = longTailConditionAligner( value, expanded_targets, candidate_targets, PICOS['P'] )
                    if condition_annotations and len( getSecOrdKeys(condition_annotations) ) > 1:
                        annotation_collector.append( condition_annotations )

                if 'ei_name' in key: 
                    candidate_targets = mapping[key]
                    int_annotations = longTailInterventionAligner( value, expanded_targets, candidate_targets, PICOS['IC'] )
                    if int_annotations and len( getSecOrdKeys(int_annotations) ) > 1:
                        annotation_collector.append( int_annotations )

                if 'ei_syn' in key:
                    candidate_targets = mapping[key]
                    int_syn_annotations = longTailInterventionAligner( value, expanded_targets, candidate_targets, PICOS['IC'] )
                    if int_syn_annotations and len( getSecOrdKeys(int_syn_annotations) ) > 1:
                        annotation_collector.append( int_syn_annotations )

                print('------------------------------------------------------------------------------')
                if 'eo_name' in key:

                    # Labeler 1 - Direct match 
                    candidate_targets = mapping[key]
                    primout_annotations = longTailOutcomeAligner( value, expanded_targets, candidate_targets, PICOS['O'] )
                    print( primout_annotations )
                    if primout_annotations and len( getSecOrdKeys(primout_annotations) ) > 1:
                        annotation_collector.append( primout_annotations )

                    # Labeler 2 POS labeler
                    primout_annotations_L2 = outcomePOSaligner( expanded_targets, candidate_targets, PICOS['O'], ['NOUN', 'PROPN', 'VERB', 'ADJ'] )
                    print( primout_annotations_L2 )
                print('------------------------------------------------------------------------------')

                # if 'eo_secondary' in key:

                #     # Labeler 1 - Direct match 
                #     candidate_targets = mapping[key]
                #     secondout_annotations = longTailOutcomeAligner( value, expanded_targets, candidate_targets, PICOS['O'] )
                #     if secondout_annotations and len( getSecOrdKeys(secondout_annotations) ) > 1:
                #         annotation_collector.append( secondout_annotations )


                if 'es_type' in key and 'N.A.' not in value:
                    candidate_targets = mapping[key]
                    studytype_annotations = regexAligner( [value], expanded_targets, candidate_targets, PICOS['S'] )   # direct aligner expects values as lists       
                    if studytype_annotations and len( getSecOrdKeys(studytype_annotations) ) > 1:
                        annotation_collector.append( studytype_annotations )


            # Direct global aggregation 
            globally_aggregated = global_annot_aggregator(annotation_collector)

            # Resolve the overlapping labels
            globally_merged = merge_labels(globally_aggregated)

            # Add CTO identifiers to the annotations
            globally_merged['id'] = NCT_id
            globally_aggregated['id'] = NCT_id

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
