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
import operator
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from functools import reduce
from itertools import chain
from random import shuffle

import numpy as np
import pandas as pd
import scipy
from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Q, Search
from nltk.tokenize import WhitespaceTokenizer
from pylab import *
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel

from CandGenUtilities.experiment_arguments import *
from CandGenUtilities.labeler_utilities import *
from CandGenUtilities.source_target_mapping import *
from LabelingFunctions.externalmodelLF import ExternalModelLabelingFunction
from LabelingFunctions.heuristicLF import heurPattern_pa, posPattern_i
from LabelingFunctions.LFutils import (label_lf_partitions,
                                       label_ont_and_write,
                                       label_umls_and_write, spansToLabels)
from LabelingFunctions.ontologyLF import *
from load_data import loadEBMPICO
from metrics.analysis import lf_coverages, lf_summary
from Ontologies.ontologyLoader import *
from Ontologies.ontoUtils import *

################################################################################
# Initialize and set seed
################################################################################
# Get the experiment arguments
args = getArguments()

# XXX Initialize LabelModel with correct cardinality
label_model = LabelModel(cardinality=2, verbose=True)

# Set seed
seed = 0
seed_everything(seed)
print('The random seed is set to: ', seed)

try:

    umls_db = '/mnt/nas2/data/systematicReview/UMLS/english_subset/umls_preprocessed/umls_meta.db'
    
    print('Retrieving UMLS ontology arm (Preprocessing applied)')
    umls_p  = loadUMLSdb(umls_db, 'P')    
    umls_i = loadUMLSdb(umls_db, 'I')
    umls_o = loadUMLSdb(umls_db, 'O')
 
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
    validation_p_labels_flatten = list(map(int, validation_p_labels_flatten))
    validation_p_labels_flatten = [-1 if x==0 else x for x in validation_p_labels_flatten]

    validation_i_labels_flatten = [item for sublist in list(validation['i']) for item in sublist]
    validation_i_labels_flatten = list(map(int, validation_i_labels_flatten))
    validation_i_labels_flatten = [-1 if x==0 else x for x in validation_i_labels_flatten]

    validation_o_labels_flatten = [item for sublist in list(validation['o']) for item in sublist]
    validation_o_labels_flatten = list(map(int, validation_o_labels_flatten))
    validation_o_labels_flatten = [-1 if x==0 else x for x in validation_o_labels_flatten]

    text = ' '.join(validation_token_flatten)
    assert len(re.split(' ', text)) == len(validation_token_flatten) == len( list(WhitespaceTokenizer().span_tokenize(text)) )
    spans = list(WhitespaceTokenizer().span_tokenize(text))

    start_spans = dict()
    for i, y in  enumerate(spans):
        ss = y[0]
        es = y[1]

        for value in range( ss, es+1 ):
            start_spans[value] = i

    # Rank the ontology based on coverage on the validation set
    # Combine the ontologies into labeling functions
    # ranked_umls_p, partitioned_umls_p = rankSAB( umls_p, 'p' )
    # ranked_umls_i, partitioned_umls_i = rankSAB( umls_i, 'i' )
    # ranked_umls_o, partitioned_umls_o = rankSAB( umls_o, 'o' )

    '''#########################################################################################
    # Level 1 - UMLS LF's - Partition function
    #########################################################################################'''
    # Label validation set using the partitions and write them to files
    # label_lf_partitions( partitioned_umls_p, umls_p, picos='p', text=text, token_flatten=validation_token_flatten, spans=spans, start_spans=start_spans)
    # label_lf_partitions( partitioned_umls_i, umls_i, picos='i', text=text, token_flatten=validation_token_flatten, spans=spans, start_spans=start_spans)
    # label_lf_partitions( partitioned_umls_o, umls_o, picos='o', text=text, token_flatten=validation_token_flatten, spans=spans, start_spans=start_spans)

    '''#########################################################################################
    # Level 1 - UMLS LF's - No partition, individual SAB LF's
    #########################################################################################'''
    for m in ['fuzzy', 'direct']:
        indir_umls = f'/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation/UMLS/{m}'
        # label_umls_and_write(indir_umls, umls_p, picos='p', text=text, token_flatten=validation_token_flatten, spans=spans, start_spans=start_spans)
        # label_umls_and_write(indir_umls, umls_i, picos='i', text=text, token_flatten=validation_token_flatten, spans=spans, start_spans=start_spans)
        # label_umls_and_write(indir_umls, umls_o, picos='o', text=text, token_flatten=validation_token_flatten, spans=spans, start_spans=start_spans)

    
    '''#########################################################################################
    # Level 2 - Non-UMLS LF's (non-UMLS Ontology labeling)
    #########################################################################################'''
    for m in ['fuzzy', 'direct']:

        indir_non_umls = f'/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation/nonUMLS/{m}'
        # for ontology, ont_name in zip([p_DO, p_DO_syn, p_ctd, p_ctd_syn], ['DO', 'DO_syn', 'CTD', 'CTD_syn'] ) :
        #     label_ont_and_write( indir_non_umls, ontology, picos='P', text=text, token_flatten=validation_token_flatten, spans=spans, start_spans=start_spans, ontology_name=ont_name  )

        # for ontology, ont_name in zip([i_ctd, i_ctd_syn, i_chebi, i_chebi_syn], ['CTD', 'CTD_syn', 'chebi', 'chebi_syn'] ) :
        #     label_ont_and_write( indir_non_umls, ontology, picos='I', text=text, token_flatten=validation_token_flatten, spans=spans, start_spans=start_spans, ontology_name=ont_name  )

        # for ontology, ont_name in zip([o_oae, o_oae_syn ], ['oae', 'oae_syn'] ) :
        #     label_ont_and_write( indir_non_umls, ontology, picos='O', text=text, token_flatten=validation_token_flatten, spans=spans, start_spans=start_spans, ontology_name=ont_name  )

    '''#########################################################################################
    # Level 3 - Distant Supervision and hand-crafted dictionary LF's
    #########################################################################################'''
    for m in ['fuzzy', 'direct']:
        indir_ds = f'/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation/DS/{m}'
        # for ontology, entity, ont_name in zip([ds_participant, ds_intervention, ds_intervention_syn, ds_outcome], ['P', 'I', 'I', 'O'], ['ds_participant', 'ds_intervetion', 'ds_intervention_syn', 'ds_outcome'] ) :
            # label_ont_and_write( indir_ds, ontology, picos=entity, text=text, token_flatten=validation_token_flatten, spans=spans, start_spans=start_spans, ontology_name=ont_name  )
    
    # Dictionary Labeling Function and Abbreviation dictionary Labeling function
    for ontology, entity, ont_name in zip([p_genders, i_comparator, p_abb], ['P', 'I', 'P'], ['dict_gender', 'dict_comparator', 'dict_p_abb'] ) : 
        indir_dict = '/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation/dictionary/direct'
        # label_ont_and_write( indir_dict, ontology, picos=entity, text=text, token_flatten=validation_token_flatten, spans=spans, start_spans=start_spans, ontology_name=ont_name  )
    

    '''#########################################################################################
    # Level 4 - Rule based LF's (ReGeX, Heuristics, Ontology based fuzzy bigram match)
    #########################################################################################'''
    # ReGeX Labeling Function
    for ontology, entity, ont_name in zip([p_sampsize, p_agerange, p_agemax], ['P', 'P', 'P'], ['regex_sampsize', 'regex_agerange', 'regex_agemax'] ) : 
        indir_reg = '/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation/heuristics/direct'
        # label_ont_and_write( indir_reg, [ontology], picos=entity, text=text, token_flatten=validation_token_flatten, spans=spans, start_spans=start_spans, ontology_name=ont_name  )
    

    # Heutistic Labeling Function
    i_posreg_labels = posPattern_i( text, validation_token_flatten, validation_pos_flatten, spans, start_spans, picos='I' )
    indir_posreg = '/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation/heuristics/direct'
    df = pd.DataFrame( {'tokens': validation_token_flatten, str('i_posreg'): i_posreg_labels })
    filename = 'lf_' + str('i_posreg') + '.tsv'
    # df.to_csv(f'{indir_posreg}/I/{filename}', sep='\t')

    pa_regex_heur_labels = heurPattern_pa( text, validation_token_flatten, validation_pos_flatten, spans, start_spans, picos='I' )
    indir_heur = '/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation/heuristics/direct'
    df = pd.DataFrame( {'tokens': validation_token_flatten, str('pa_regex_heur'): pa_regex_heur_labels })
    filename = 'lf_' + str('pa_regex_heur') + '.tsv'
    # df.to_csv(f'{indir_heur}/P/{filename}', sep='\t')

    '''#########################################################################################
    # TODO  Level 5 - External Model Labeling function
    #########################################################################################'''
    ExternalModelLabelingFunction()

    '''
    #########################################################################################
    # Combine LF's into a single LF
    #########################################################################################
    # L_p = [umls_p_labels, p_DO_labels, p_DO_syn_labels, p_ctd_labels, p_ctd_syn_labels, p_DS_labels, gender_labels, p_abb_labels, samplesize_labels, agerange_labels, agemax_labels, pa_regex_heur_labels, umls_p_fz_labels, p_DO_fz_labels, p_DO_syn_fz_labels, p_ctd_fz_labels, p_ctd_syn_fz_labels, p_DS_fz_labels ]
    L_i = [umls_i_labels, i_ctd_labels, i_ctd_syn_labels, i_chebi_labels, i_chebi_syn_labels, i_ds_labels, i_syn_ds_labels, comparator_labels, i_posreg_labels, umls_i_fz_labels, i_ctd_fz_labels, i_ctd_syn_fz_labels, i_chebi_fz_labels, i_chebi_syn_fz_labels, i_ds_fz_labels, i_syn_ds_fz_labels ]
    # L_o = [umls_o_labels, o_oae_labels, o_oae_syn_labels, o_ds_labels, umls_o_fz_labels, o_oae_fz_labels, o_oae_syn_fz_labels, o_ds_fz_labels ]
    '''

    # L_p = scipy.sparse.csr_matrix( L_p )
    # participant_LF_summary = lf_summary(L_p, Y=validation_p_labels_flatten)
    # print( participant_LF_summary )

    # start_time = time.time()
    # print('Training Label Model...')
    # L = np.array(L_i)
    # label_model.fit(L, seed=seed, n_epochs=100)
    # print("--- %s seconds ---" % (time.time() - start_time))

    # Y_hat = label_model.predict_proba(L)

    # print( type(Y_hat) )

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