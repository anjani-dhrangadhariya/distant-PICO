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

from SourceTargetAligner.source_target_mapping import generateMapping
from SourceTargetAligner.align_pariticipants import *

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
# for hit in results_gen: # XXX: Entire CTO
for n, hit in enumerate( res['hits']['hits'] ): # XXX: Only a part search results from the CTO

    write_hit = collections.defaultdict(dict) # Dictionary to store the annotations for the CTO record being iterated

    fullstudy = hit['_source']['FullStudiesResponse']['FullStudies'][0]['Study']
    NCT_id = hit['_source']['FullStudiesResponse']['Expression']
    write_hit['id'] = NCT_id

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
        # print(mapping)
        # print(NCT_id , ': ', expanded_targets.keys())
        # print( expanded_sources.keys() )


        # XXX: Direct matching begins
        for key, value in expanded_sources.items():
            
            # only get the gender values
            if 'gender' in key:
                candidate_targets = mapping[key]
                gender_annotations = alignParGender( value, expanded_targets, candidate_targets )
                if gender_annotations:    
                    print( gender_annotations )

            # only get the gender values
            # if 'sample_size' in key:
            #     candidate_targets = mapping[key]
            #     sampsize_annotations = alignParSampSize( value, expanded_targets, candidate_targets )          
            #     print( sampsize_annotations )           


    except:
        logNCTID = 'Caused exception at the NCT ID: ' + NCT_id
        logging.info(logNCTID)