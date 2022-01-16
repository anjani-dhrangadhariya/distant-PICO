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

# Import DISANT-PICO modules
from CandGenUtilities.experiment_arguments import *
from CandGenUtilities.labeler_utilities import *
from CandGenUtilities.source_target_mapping import *
from SourceFetcher.int_sourcefetcher import *
from SourceFetcher.outcome_sourcefetcher import *
from SourceFetcher.parti_sourcefetcher import *
from SourceFetcher.stdtype_sourcefetcher import *
from SourceTargetExpander.expand_sources import *
from SourceTargetExpander.expand_targets import *
from TargetFetcher.all_targetsfetcher import *

################################################################################
# Initialize 
################################################################################
# Get the experiment arguments
args = getArguments()

################################################################################
# Instantiate ElasticSearch
################################################################################
es = Elasticsearch( [{u'host': u'127.0.0.1', u'port': b'9200'}] )

outdir = '/mnt/nas2/data/systematicReview/ds_cto_dict'

def writeDSsources(data, fileHandle):
    if data:
        json.dump(data, fileHandle)
        fileHandle.write('\n')

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

res = es.search(index="ctofull-index", body={"query": {"match_all": {}}}, size=100)
print('Total number of records retrieved: ', res['hits']['total']['value'])

with open(f'{outdir}/intervention.txt', 'w') as iwf,open(f'{outdir}/intervention_syn.txt', 'w') as iswf, open(f'{outdir}/outcome.txt', 'w') as owf, open(f'{outdir}/participant.txt', 'w') as pwf:


    # Iterate through all of the fetched CTO index documents
    # for n, hit in enumerate( res['hits']['hits'] ): # Only a part search results from the CTO
    for hit in results_gen: # Entire CTO

        hit_tokens = []
        hit_l1l2_labels = []
        sentence_mapper = []

        try:

            fullstudy = hit['_source']['FullStudiesResponse']['FullStudies'][0]['Study']
            NCT_id = hit['_source']['FullStudiesResponse']['Expression']

            protocol_section = fullstudy['ProtocolSection']
            derieved_section = fullstudy['DerivedSection']

            #############################################################################
            # Retrieve the distant supervision sources of PICOS annotation (TODO: This is some form of ETL if multiple data sources were involved!!!)
            #############################################################################
            participants = fetchParticipantSources(protocol_section)
            expanded_condition = expandCondition(participants['p_condition'], fetch_pos=True, fetch_abb=True)
            writeDSsources(expanded_condition, pwf)

            intervention_comparator, intervention_comparator_syn = fetchIntcompSources(protocol_section)
            expanded_intervention = expandIntervention(intervention_comparator, fetch_pos=True, fetch_abb=True)
            writeDSsources(expanded_intervention, iwf)
            expanded_intervention_syn = expandIntervention(intervention_comparator_syn, fetch_pos=True, fetch_abb=True)
            writeDSsources(expanded_intervention_syn, iswf)

            outcomes = fetchOutcomeSources(protocol_section)
            expanded_prim_outcomes = expandOutcomes(outcomes, fetch_pos = True, fetch_abb = True)
            writeDSsources(expanded_prim_outcomes, owf)

            study_type = fetchStdTypeSources(protocol_section) # TODO: Expand the study type sources

            log_success = 'Wrote the weak annotations for ' + str(NCT_id)
            # print(log_success)
            logging.info(log_success)  

        except Exception as ex:
          
            template = "An exception of type {0} occurred. Arguments:{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print( NCT_id , ' : ', message )

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

            print(traceback.format_exc())

            logging.info(NCT_id)
            logging.info(message)
            string2log = str(exc_type) + ' : ' + str(fname) + ' : ' + str(exc_tb.tb_lineno)
            logging.info(string2log)
            logging.info(traceback.format_exc())