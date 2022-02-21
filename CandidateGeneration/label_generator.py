#!/usr/bin/env python

def SourceTargetsDoc(a):
    '''The module generates distant annotations for PICO entities using a cominbation of distant supervision and dynamic programming'''
    return a**a

print( SourceTargetsDoc.__doc__ )

from asyncore import write
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
import sys, argparse
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
from LabelingFunctions.heuristicLF import heurPattern_p_sampsize, heurPattern_pa, posPattern_i
from LabelingFunctions.LFutils import (label_lf_partitions,
                                       label_ont_and_write,
                                       label_umls_and_write, spansToLabels)
from LabelingFunctions.ontologyLF import *
from load_data import loadEBMPICO
from Ontologies.ontologyLoader import *
from Ontologies.ontoUtils import *

################################################################################
# Initialize and set seed
################################################################################
# XXX Initialize LabelModel with correct cardinality
label_model = LabelModel(cardinality=2, verbose=True)

# Set seed
seed = 0
seed_everything(seed)
print('The random seed is set to: ', seed)

# Parse arguments for experiment flow
parser = argparse.ArgumentParser()
parser.add_argument('-level1', type=bool, default=False) # Level1 = UMLS LF's
parser.add_argument('-level2', type=bool, default=False) # Level2: Non-UMLS LF's
parser.add_argument('-level3', type=bool, default=False) # Level 3 = Distant Supervision LF's
parser.add_argument('-level4', type=bool, default=False) # Level 4 = Rule based LF's (ReGeX, Heuristics and handcrafted dictionaries)
parser.add_argument('-level5', type=bool, default=False) # Level 5 = External Model LF's
parser.add_argument('-levels', type=bool, default=False) # execute data labeling using all levels
parser.add_argument('-umls_fpath', type=Path, default= 'UMLS/english_subset/umls_preprocessed/umls_tui_pio2.db')
parser.add_argument('-ds_fpath', type=Path, default='/data/systematicReview/ds_cto_dict' )
parser.add_argument('-indir', type=Path, default='/data/systematicReview' ) # directory with labeling function sources
parser.add_argument('-outdir', type=Path, default='/results/Results/systematicReview' ) # directory path to store the weakly labeled candidates
args = parser.parse_args()

try:

    ##############################################################################################################
    # Load EBM-PICO training data
    ################################################################################
    ebm_nlp = '/mnt/nas2/data/systematicReview/PICO_datasets/EBM_parsed'
    text, df_data_token_flatten, df_data_pos_flatten, df_data_p_labels_flatten, df_data_i_labels_flatten, df_data_o_labels_flatten = loadEBMPICO( ebm_nlp, write_to_file = False )
    spans = list(WhitespaceTokenizer().span_tokenize(text))

    start_spans = dict()
    for i, y in  enumerate(spans):
        ss = y[0]
        es = y[1]

        for value in range( ss, es+1 ):
            start_spans[value] = i

    '''#########################################################################################
    # Level 1 - UMLS LF's
    #########################################################################################'''
    if args.level1 == True or args.levels == True:

        ''' umls_tui_pio2 is used to generate UMLS2 candidate annotations: /systematicReview/distant_pico/candidate_generation/UMLS2
        # umls_v2.db is used to generate UMLS candidate annotations: /systematicReview/distant_pico/candidate_generation/UMLS
        '''
        umls_db = f'{args.indir}/{args.umls_fpath}'
        print('Retrieving UMLS ontology arm (Preprocessing applied)')
        umls_p  = loadUMLSdb(umls_db, entity='P')
        umls_i = loadUMLSdb(umls_db, entity='I')
        umls_o = loadUMLSdb(umls_db, entity='O')

        for m in ['fuzzy', 'direct']: # fuzzy = fuzzy bigram match, direct = no fuzzy bigram match
            outdir_umls = f'{args.outdir}/distant_pico/candidate_generation/UMLS2/{m}'
            for entity, umls_d in zip(['p', 'i', 'o'], [umls_p, umls_i, umls_o]):
                label_umls_and_write(outdir_umls, umls_d, picos=entity, text=text, token_flatten=df_data_token_flatten, spans=spans, start_spans=start_spans, write = False)


    '''#########################################################################################
    # Level 2 - Non-UMLS LF's (non-UMLS Ontology labeling)
    #########################################################################################'''
    if args.level2 == True or args.levels == True:

        print('Retrieving non-UMLS Ontologies  (Preprocessing applied)')
        p_DO, p_DO_syn = loadOnt( f'{args.indir}/Ontologies/participant/DOID.csv', delim = ',', term_index = 1, term_syn_index = 2  )
        p_ctd, p_ctd_syn = loadOnt( f'{args.indir}/Ontologies/participant/CTD_diseases.tsv', delim = '\t', term_index = 0, term_syn_index = 7 )
        p_HPO, p_HPO_syn = loadOnt( f'{args.indir}/Ontologies/participant/HP.csv', delim = ',', term_index = 1, term_syn_index = 2  )
        i_ctd, i_ctd_syn = loadOnt( f'{args.indir}/Ontologies/intervention/CTD_chemicals.tsv', delim = '\t', term_index = 0, term_syn_index = 7 )
        i_chebi, i_chebi_syn = loadOnt( f'{args.indir}/Ontologies/intervention/CHEBI.csv', delim = ',', term_index = 1, term_syn_index = 2  )
        o_oae, o_oae_syn = loadOnt( f'{args.indir}/Ontologies/outcome/OAE.csv', delim=',', term_index=1, term_syn_index=2 )

        for m in ['fuzzy', 'direct']:
            outdir_non_umls = f'{args.outdir}/distant_pico/candidate_generation/nonUMLS/{m}'
            for ontology, ont_name in zip([p_HPO, p_HPO_syn, p_DO, p_DO_syn, p_ctd, p_ctd_syn], ['HPO', 'HPO_syn', 'DO', 'DO_syn', 'CTD', 'CTD_syn'] ) :
               nonUMLS_p_labels = label_ont_and_write( outdir_non_umls, ontology, picos='P', text=text, token_flatten=df_data_token_flatten, spans=spans, start_spans=start_spans, write = False, ontology_name=ont_name, expand_term=True)

            for ontology, ont_name in zip([i_ctd, i_ctd_syn, i_chebi, i_chebi_syn], ['CTD', 'CTD_syn', 'chebi', 'chebi_syn'] ) :
                nonUMLS_i_labels = label_ont_and_write( outdir_non_umls, ontology, picos='I', text=text, token_flatten=df_data_token_flatten, spans=spans, start_spans=start_spans, write = False, ontology_name=ont_name, expand_term=True )

            for ontology, ont_name in zip([o_oae, o_oae_syn ], ['oae', 'oae_syn'] ) :
                nonUMLS_o_labels = label_ont_and_write( outdir_non_umls, ontology, picos='O', text=text, token_flatten=df_data_token_flatten, spans=spans, start_spans=start_spans, write = False, ontology_name=ont_name, expand_term=True )

    '''#########################################################################################
    # Level 3 - Distant Supervision LF's
    #########################################################################################'''
    if args.level3 == True or args.levels == True:

        print('Retrieving distant supervision dictionaries')
        ds_participant = loadDS(args.ds_fpath, 'participant')
        ds_intervention = loadDS(args.ds_fpath, 'intervention')
        ds_intervention_syn = loadDS(args.ds_fpath, 'intervention_syn')
        ds_outcome = loadDS(args.ds_fpath, 'outcome')

        for m in ['fuzzy', 'direct']:
            outdir_ds = f'{args.outdir}/distant_pico/candidate_generation/DS/{m}'
            for ontology, entity, ont_name in zip([ds_participant], ['P'], ['ds_participant'] ) :
                ds_p_labels = label_ont_and_write( outdir_ds, ontology, picos=entity, text=text, token_flatten=df_data_token_flatten, spans=spans, start_spans=start_spans, write=False, ontology_name=ont_name, expand_term=True  )

            for ontology, entity, ont_name in zip([ds_intervention, ds_intervention_syn], ['I', 'I'], ['ds_intervetion', 'ds_intervention_syn'] ) :
                ds_i_labels = label_ont_and_write( outdir_ds, ontology, picos=entity, text=text, token_flatten=df_data_token_flatten, spans=spans, start_spans=start_spans, write=False, ontology_name=ont_name, expand_term=True  )
        
            for ontology, entity, ont_name in zip([ds_outcome], ['O'], ['ds_outcome'] ) :
                ds_o_labels = label_ont_and_write( outdir_ds, ontology, picos=entity, text=text, token_flatten=df_data_token_flatten, spans=spans, start_spans=start_spans, write=False, ontology_name=ont_name, expand_term=True  )
    

    '''##############################################################################################################
    # Level 4 - Rule based LF's (ReGeX, Heuristics and handcrafted dictionaries)
    ##############################################################################################################'''
    if args.level4 == True or args.levels == True:

        print('Retrieving ReGeX patterns')
        p_sampsize = loadPattern( 'samplesize' )
        p_sampsize2 = loadPattern( 'samplesize2' )
        p_agerange = loadPattern( 'age1' )
        p_agemax = loadPattern( 'age2' )
        p_meanage = loadPattern( 'meanage' )
        s_study_type = loadPattern( 'studytype' )

        # ReGeX Labeling Function
        for reg_lf_i, entity, reg_lf_name in zip([p_sampsize, p_sampsize2, p_agerange, p_agemax, p_meanage, s_study_type], ['P', 'P', 'P', 'P', 'P', 'S'], ['regex_sampsize', 'regex_sampsize2', 'regex_agerange', 'regex_agemax', 'regex_meanage', 'regex_stdtype'] ) : 
            outdir_reg = f'{args.outdir}/distant_pico/candidate_generation/heuristics/direct'
            regex_labels = label_ont_and_write( outdir_reg, [reg_lf_i], picos=entity, text=text, token_flatten=df_data_token_flatten, spans=spans, start_spans=start_spans, write=False, ontology_name=reg_lf_name, expand_term=False )

        outdir_heurPattern = f'{args.outdir}/distant_pico/candidate_generation/heuristics/direct'
        
        # Heutistic Labeling Functions
        i_posreg_labels = posPattern_i( text, df_data_token_flatten, df_data_pos_flatten, spans, start_spans, picos='I' )
        df = pd.DataFrame( {'tokens': df_data_token_flatten, str('i_posreg'): i_posreg_labels })
        filename = 'lf_' + str('i_posreg') + '.tsv'
        df.to_csv(f'{outdir_heurPattern}/I/{filename}', sep='\t')

        pa_regex_heur_labels = heurPattern_pa( text, df_data_token_flatten, df_data_pos_flatten, spans, start_spans, picos='P' )
        df = pd.DataFrame( {'tokens': df_data_token_flatten, str('pa_regex_heur'): pa_regex_heur_labels })
        filename = 'lf_' + str('pa_regex_heur') + '.tsv'
        df.to_csv(f'{outdir_heurPattern}/P/{filename}', sep='\t')

        ps_heurPattern_labels = heurPattern_p_sampsize( text, df_data_token_flatten, df_data_pos_flatten, spans, start_spans, picos='P' )
        df = pd.DataFrame( {'tokens': df_data_token_flatten, str('ps_heurPattern_labels'): ps_heurPattern_labels })
        filename = 'lf_' + str('ps_heurPattern_labels') + '.tsv'
        df.to_csv(f'{outdir_heurPattern}/P/{filename}', sep='\t')
        
        print('Retrieving hand-crafted dictionaries')
        p_genders = loadDict(f'{args.indir}/Ontologies/participant/gender_sexuality.txt')
        i_comparator = loadDict(f'{args.indir}/Ontologies/intervention/comparator_dict.txt')
        o_endpoints = loadDict(f'{args.indir}/Ontologies/outcome/endpoints_dict.txt')

        print('Retrieving abbreviations dictionaries')  
        p_abb = loadAbbreviations(f'{args.indir}/Ontologies/participant/diseaseabbreviations.tsv')

        # Dictionary Labeling Function and Abbreviation dictionary Labeling function
        for ontology, entity, ont_name in zip([p_genders, i_comparator, p_abb, o_endpoints], ['P', 'I', 'P', 'O'], ['dict_gender', 'dict_comparator', 'dict_p_abb', 'dict_o_terms'] ) : 
            outdir_dict = f'{args.outdir}/distant_pico/candidate_generation/dictionary/direct'
            label_ont_and_write( outdir_dict, ontology, picos=entity, text=text, token_flatten=df_data_token_flatten, spans=spans, start_spans=start_spans, write=False, ontology_name=ont_name, expand_term=False  )

    '''#########################################################################################
    # TODO  Level 5 - External Model Labeling function
    #########################################################################################'''
    if args.level5 == True or args.levels == True:
        # TODO: Retrieve external models
        ExternalModelLabelingFunction()

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