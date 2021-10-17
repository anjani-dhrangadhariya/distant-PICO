#!/usr/bin/env python

def SourceTargetsDoc(a):
    '''The module generates distant annotations for PICO entities using a cominbation of distant supervision and dynamic programming'''
    return a**a

print( SourceTargetsDoc.__doc__ )

__author__ = "Anjani Dhrangadhariya"
__maintainer__ = "Anjani Dhrangadhariya"
__email__ = "anjani.dhrangadhariya@hevs.ch"
__status__ = "Prototype"

import sys, json, os
import logging
import datetime as dt
import time
import random 

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search,  Q

import difflib, re

from collections import Counter
from collections import defaultdict
import collections
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pylab import *

# Import DISANT-PICO modules
from SourceFetcher.parti_sourcefetcher import *
from SourceFetcher.int_sourcefetcher import *
from SourceFetcher.outcome_sourcefetcher import *
from SourceFetcher.stdtype_sourcefetcher import *

from TargetFetcher.all_targetsfetcher import *

from SourceTargetExpander.expand_sources import *
from SourceTargetExpander.expand_targets import *

from SourceTargetAligner.source_target_mapping import *
from SourceTargetAligner.labeling_operators import *

from AnnotationAggregation.label_aggregator import *

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

res = es.search(index="ctofull2021-index", body={"query": {"match_all": {}}}, size=6000)
print('Total number of records retrieved: ', res['hits']['total']['value'])


# Iterate through all of the fetched CTO index documents

theFile ='/mnt/nas2/data/systematicReview/clinical_trials_gov/distant_pico_pre/secondary_outcomes.txt'
with open(theFile, 'a+') as wf:

    for n, hit in enumerate( res['hits']['hits'] ): # Only a part search results from the CTO
    # for hit in results_gen: # Entire CTO
        write_hit = collections.defaultdict(dict) # Dictionary to store the annotations for the CTO record being iterated

        fullstudy = hit['_source']['FullStudiesResponse']['FullStudies'][0]['Study']
        NCT_id = hit['_source']['FullStudiesResponse']['Expression']
        write_hit['id'] = NCT_id

        # Annotation aggregator for PICOS for the NCT_id
        p_aggregator = dict()
        ic_aggregator = dict()
        o_aggregator = dict()
        s_aggregator = dict()
        # p_aggregator['id'] = NCT_id
        # ic_aggregator['id'] = NCT_id
        # o_aggregator['id'] = NCT_id
        # s_aggregator['id'] = NCT_id

        try:

            protocol_section = fullstudy['ProtocolSection']
            derieved_section = fullstudy['DerivedSection']

            # Retrieve the sources of PICOS annotation
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

            # XXX: How to adjust for the abbreviation detection? Decide this based on direct matching process

            # Get the mappings between sources and their relevant targets
            mapping = generateMapping()

            # Get the mappings between PICOS entity and their labels
            PICOS = generateLabels()

            # Add all target keys to the label aggregator
            # p_aggregator = dict.fromkeys(expanded_targets.keys(),[])
            # ic_aggregator = dict.fromkeys(expanded_targets.keys(),[])
            # o_aggregator = dict.fromkeys(expanded_targets.keys(),[])
            # s_aggregator = dict.fromkeys(expanded_targets.keys(),[])

            #################################################################
            # Direct matching begins
            # P = 1, I/C = 2, O = 3, S = 4
            # expanded_sources: All the sources from a NCTID; expanded_targets: All the targets from a NCTID
            #################################################################           
            for key, value in expanded_sources.items():

                if 'ei_name' in key: 
                    candidate_targets = mapping[key]
                    int_annotations = longTailInterventionAligner( value, expanded_targets, candidate_targets, PICOS['IC'] )
                    if int_annotations:
                        ic_aggregator = aggregate_labels(int_annotations, ic_aggregator)
                        # print('Intervention annotations')
                        # print( int_annotations )

                if 'ei_syn' in key:
                    candidate_targets = mapping[key]
                    int_syn_annotations = longTailInterventionAligner( value, expanded_targets, candidate_targets, PICOS['IC'] )
                    if int_syn_annotations:
                        i_aggregated = aggregate_labels(int_syn_annotations, ic_aggregator)
                        # print('Intervention synonyms annotations')  
                        # print( int_syn_annotations )

                if 'gender' in key:
                    candidate_targets = mapping[key]
                    gender_annotations = directAligner( value, expanded_targets, candidate_targets, PICOS['P'] )
                    if gender_annotations:
                        p_aggregator = aggregate_labels(gender_annotations, p_aggregator)
                    #     print('Gender annotations')
                    #     print( gender_annotations )

                if 'sample_size' in key:
                    candidate_targets = mapping[key]
                    sampsize_annotations = directAligner( [value], expanded_targets, candidate_targets, PICOS['P'] )   # direct aligner expects values as lists       
                    if sampsize_annotations:  
                        p_aggregator = aggregate_labels(sampsize_annotations, p_aggregator)
                    #     print('Sample size annotations')
                    #     print( sampsize_annotations )

                if 'age' in key:
                    candidate_targets = mapping[key]
                    stdage_annotations = directAligner( value['StdAge'], expanded_targets, candidate_targets, PICOS['P'] )
                    if stdage_annotations:
                        p_aggregator = aggregate_labels(stdage_annotations, p_aggregator)
                    #     print('Std age annotations')
                    #     print( stdage_annotations )

                    exctage_annotattions = regexAligner( [value['exactAge']], expanded_targets, candidate_targets, PICOS['P'] ) # reGeX aligner expects values as lists   
                    # if exctage_annotattions: 
                    #     print('Exact age annotations')
                    #     print( exctage_annotattions )
                    

                if 'condition' in key:
                    candidate_targets = mapping[key]
                    condition_annotations = longTailConditionAligner( value, expanded_targets, candidate_targets, PICOS['P'] )
                    # if condition_annotations:  
                    #     print('Condition annotations')
                    #     print( condition_annotations )


                if 'es_type' in key:
                    candidate_targets = mapping[key]
                    studytype_annotations = regexAligner( [value], expanded_targets, candidate_targets, PICOS['S'] )   # direct aligner expects values as lists       
                    # if studytype_annotations:
                    #     print('Study type annotations')
                    #     print( studytype_annotations )

                if 'eo_primary' in key:
                    candidate_targets = mapping[key]
                    primout_annotations = longTailOutcomeAligner( value, expanded_targets, candidate_targets, PICOS['O'] )
                    # if primout_annotations:
                    #     print('Primary outcomes annotations')
                    #     print( primout_annotations )

                if 'eo_secondary' in key:
                    candidate_targets = mapping[key]
                    secondout_annotations = longTailOutcomeAligner( value, expanded_targets, candidate_targets, PICOS['O'] )
                    # if secondout_annotations:
                    #     print('Secondary outcomes annotations')
                    #     print( secondout_annotations )

            print('###########################################################################')


        except:

            logNCTID = 'Caused exception at the NCT ID: ' + NCT_id
            logging.info(logNCTID)