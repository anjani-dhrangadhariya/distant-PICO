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

# Retrieve the UMLS arm of PICOS annotation
loadUMLSdb('/mnt/nas2/data/systematicReview/UMLS/english_subset/umls_preprocessed/umls.db')



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

res = es.search(index="ctofull2021-index", body={"query": {"match_all": {}}}, size=5)
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
            intervention_comparator = fetchIntcompSources(protocol_section)
            outcomes = fetchOutcomeSources(protocol_section)
            study_type = fetchStdTypeSources(protocol_section)

            sources = {**participants, **intervention_comparator, **outcomes, **study_type}


            # Scenario I.Weak labeling: Distant Supervision
            expanded_sources_i = expandSources_i(protocol_section, sources)

            # Scenario II. Weak labeling: Distant Supervision + Task-specific rules
            # expanded_sources_ii = expandSources_ii(protocol_section, sources) 

            # Retrieve the targets of PICOS annotation
            targets = fetchTargets(protocol_section)

            # Expand the targets of PICOS annotation
            expanded_targets = expandTargets(protocol_section, targets)

            #################################################################
            # Direct matching begins
            # P = 1, I/C = 2, O = 3, S = 4, ABSTAIN = -1, Ospan = 0
            # expanded_sources: All the sources from entire CTO NCTIDs; expanded_targets: All the targets from entire CTO NCTIDs
            #################################################################  
            o_annotation_collector = []
            annotation_collector = []

            # Scenario I.Weak labeling: Ontology
            # scenario_1_annotations = scheme_i( expanded_sources_i, expanded_targets, PICOS, abstain_options )

            # Scenario II. Weak labeling: Ontology + (Distant Supervision + Task-specific rules)
            # scenario_2_annotations = scheme_ii( sources, expanded_targets, PICOS )

            '''
            for key, value in expanded_sources.items():

                if 'gender' in key:
                    candidate_targets = mapping[key]
                    gender_annotations = directAligner( value, expanded_targets, candidate_targets, PICOS['P'] )
                    if gender_annotations and len( getSecOrdKeys(gender_annotations) ) > 1:
                        annotation_collector.append( gender_annotations )


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

                if 'eo_name' in key:
                  
                    if args.o_labeler1 == True:
                        # Labeler 1 - Direct match
                        candidate_targets = mapping[key]
                        out_annotations_L1 = longTailOutcomeAligner( value, expanded_targets, candidate_targets, PICOS['O'] )
                        if out_annotations_L1 and len( getSecOrdKeys(out_annotations_L1) ) > 1:
                            out_annotations_L1 = merge_sources( out_annotations_L1 )
                            o_annotation_collector.append( out_annotations_L1 )

                    if args.o_labeler2 == True:
                        # Labeler 2 POS labeler
                        out_annotations_L2 = outcomePOSaligner( expanded_targets, candidate_targets, PICOS['O'], allowedPOS() )
                        if out_annotations_L2:
                            o_annotation_collector.append( out_annotations_L2 )


                if 'es_type' in key and 'N.A.' not in value:
                    candidate_targets = mapping[key]
                    studytype_annotations = regexAligner( [value], expanded_targets, candidate_targets, PICOS['S'] )   # direct aligner expects values as lists       
                    if studytype_annotations and len( getSecOrdKeys(studytype_annotations) ) > 1:
                        annotation_collector.append( studytype_annotations )

            # All the annotations for individual entities are collected here
            if o_annotation_collector:
                # print( o_annotation_collector )
                for lf1, lf2 in zip( o_annotation_collector[0], o_annotation_collector[1] ):
                    # print( lf1, ' : ', lf2 )
                    l1 = o_annotation_collector[0][lf1]
                    l2 = o_annotation_collector[1][lf2]

                    for sentence_l1, sentence_l2 in zip( l1, l2 ):
                        common_tokens = l1[sentence_l1]['tokens']
                        l1_labels = l1[sentence_l1][str(3)]
                        l2_labels = l2[sentence_l2][str(3)]

                        if set(l1_labels) != -1 and set(l2_labels) != -1:
                            hit_tokens.extend( common_tokens )
                            for wl1, wl2 in zip( l1_labels, l2_labels ):
                                hit_l1l2_labels.append( [ int(wl1), int(wl2), int(3) ] )
            '''


            # Direct global aggregation 
            # globally_aggregated = global_annot_aggregator(annotation_collector)

            # # Resolve the overlapping labels
            # globally_merged = merge_labels(globally_aggregated)

            # # Add CTO identifiers to the annotations
            # globally_merged['id'] = NCT_id
            # globally_aggregated['id'] = NCT_id

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