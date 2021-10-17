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

res = es.search(index="ctofull2021-index", body={"query": {"match_all": {}}}, size=50)
print('Total number of records retrieved: ', res['hits']['total']['value'])


# Iterate through all of the fetched CTO index documents
theFile ='/mnt/nas2/data/systematicReview/clinical_trials_gov/distant_pico_pre/secondary_outcomes.txt'
with open(theFile, 'a+') as wf:

    for n, hit in enumerate( res['hits']['hits'] ): # Only a part search results from the CTO
    # for hit in results_gen: # Entire CTO

        try:

            print('###########################################################################')
            write_hit = collections.defaultdict(dict) # Dictionary to store the annotations for the CTO record being iterated

            fullstudy = hit['_source']['FullStudiesResponse']['FullStudies'][0]['Study']
            NCT_id = hit['_source']['FullStudiesResponse']['Expression']
            print(NCT_id)
            write_hit['id'] = NCT_id

            # Annotation aggregator for PICOS for the NCT_id
            p_aggregator = dict()
            ic_aggregator = dict()
            o_aggregator = dict()
            s_aggregator = dict()

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

            # Get the mappings between sources and their relevant targets
            mapping = generateMapping()

            # Get the mappings between PICOS entity and their labels
            PICOS = generateLabels()

            #################################################################
            # Direct matching begins
            # P = 1, I/C = 2, O = 3, S = 4
            # expanded_sources: All the sources from a NCTID; expanded_targets: All the targets from a NCTID
            #################################################################           
            for key, value in expanded_sources.items():

                if 'ei_name' in key: 
                    candidate_targets = mapping[key]
                    # int_annotations = longTailInterventionAligner( value, expanded_targets, candidate_targets, PICOS['IC'] )
                    # if int_annotations:
                        # print('Intervention annotations')
                        # print( int_annotations )
                        # ic_aggregator = aggregate_labels(int_annotations, ic_aggregator)


                if 'ei_syn' in key:
                    candidate_targets = mapping[key]
                    # int_syn_annotations = longTailInterventionAligner( value, expanded_targets, candidate_targets, PICOS['IC'] )
                    # if int_syn_annotations:
                        # print('Intervention synonyms annotations')
                        # print( int_syn_annotations )
                        # ic_aggregator = aggregate_labels(int_syn_annotations, ic_aggregator)


                if 'gender' in key:
                    candidate_targets = mapping[key]
                    # gender_annotations = directAligner( value, expanded_targets, candidate_targets, PICOS['P'] )
                    # if gender_annotations:
                    #     p_aggregator = aggregate_labels(gender_annotations, p_aggregator)
                    #     print('Gender annotations')
                    #     print( gender_annotations )

                if 'sample_size' in key:
                    candidate_targets = mapping[key]
                    # sampsize_annotations = directAligner( [value], expanded_targets, candidate_targets, PICOS['P'] )   # direct aligner expects values as lists       
                    # if sampsize_annotations:  
                    #     p_aggregator = aggregate_labels(sampsize_annotations, p_aggregator)
                    #     print('Sample size annotations')
                    #     print( sampsize_annotations )

                if 'age' in key:
                    candidate_targets = mapping[key]
                    # stdage_annotations = directAligner( value['StdAge'], expanded_targets, candidate_targets, PICOS['P'] )
                    # if stdage_annotations:
                    #     p_aggregator = aggregate_labels(stdage_annotations, p_aggregator)
                    #     print('Std age annotations')
                    #     print( stdage_annotations )

                    # exctage_annotattions = regexAligner( [value['exactAge']], expanded_targets, candidate_targets, PICOS['P'] ) # reGeX aligner expects values as lists   
                    # if exctage_annotattions: 
                    #     p_aggregator = aggregate_labels(exctage_annotattions, p_aggregator)
                    #     print('Exact age annotations')
                    #     print( exctage_annotattions )
                    

                if 'condition' in key:
                    candidate_targets = mapping[key]
                    # condition_annotations = longTailConditionAligner( value, expanded_targets, candidate_targets, PICOS['P'] )
                    # if condition_annotations:  
                    #     print('Condition annotations')
                    #     print( condition_annotations )


                if 'es_type' in key and 'N.A.' not in value:
                    candidate_targets = mapping[key]
                    studytype_annotations = regexAligner( [value], expanded_targets, candidate_targets, PICOS['S'] )   # direct aligner expects values as lists       
                    # if studytype_annotations:
                    #     print('Study type annotations')
                    #     print( studytype_annotations )
                    # s_aggregator = aggregate_labels(studytype_annotations, s_aggregator)

                if 'eo_primary' in key:
                    candidate_targets = mapping[key]
                    # primout_annotations = longTailOutcomeAligner( value, expanded_targets, candidate_targets, PICOS['O'] )
                    # if primout_annotations:
                    #     print('Primary outcomes annotations')
                    #     o_aggregator = aggregate_labels(primout_annotations, o_aggregator)
                    #     print( o_aggregator )

                if 'eo_secondary' in key:
                    candidate_targets = mapping[key]
                    # secondout_annotations = longTailOutcomeAligner( value, expanded_targets, candidate_targets, PICOS['O'] )
                    # if secondout_annotations:
                        # print('Secondary outcomes annotations')
                        # o_aggregator = aggregate_labels(secondout_annotations, o_aggregator)
                        # print( secondout_annotations )

            print( 'Final study type aggregator: ' , o_aggregator )

        except Exception as ex:
          
            template = "An exception of type {0} occurred. Arguments:{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print( NCT_id , ' : ', message )

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

            # logging.info(message)