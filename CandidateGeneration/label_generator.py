#!/usr/bin/env python

def SourceTargetsDoc(a):
    '''The module generates distant annotations for PICO entities using a cominbation of distant supervision and dynamic programming'''
    return a**a

print( SourceTargetsDoc.__doc__ )

import argparse
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
from asyncore import write
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
from LabelingFunctions.heuristicLF import (heurPattern_p_sampsize,
                                           heurPattern_pa, posPattern_i)
from LabelingFunctions.LFutils import (label_abb_and_write, label_heur_and_write, label_lf_partitions,
                                       label_ont_and_write,
                                       label_regex_and_write,
                                       label_umls_and_write, spansToLabels)
from LabelingFunctions.ontologyLF import *
from load_data import loadEBMPICO
from Ontologies.ontologyLoader import *
from Ontologies.ontoUtils import *

################################################################################
# Set seed
################################################################################
seed = 0
seed_everything(seed)
print('The random seed is set to: ', seed)

################################################################################
# Load stopwords (generic negative label LFs)
################################################################################
#sw_lf = loadStopWords()

################################################################################
# Set global variable
################################################################################
candgen_version = 'v4' # version = {v3, v4, ...}

if candgen_version == 'v3':
    if_stopwords = True
elif candgen_version == 'v4':
    if_stopwords = False

extract_abbs = False

################################################################################
# Parse arguments for experi flow
################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-level1', type=bool, default=False) # Level1 = UMLS LF's
parser.add_argument('-level2', type=bool, default=False) # Level2: Non-UMLS LF's
parser.add_argument('-level3', type=bool, default=False) # Level 3 = Distant Supervision LF's
parser.add_argument('-level4', type=bool, default=False) # Level 4 = Rule based LF's (ReGeX, Heuristics and handcrafted dictionaries)
parser.add_argument('-level5', type=bool, default=False) # Level 5 = Abbreviation LFs
parser.add_argument('-level6', type=bool, default=False) # Level 6 = External Model LF's
parser.add_argument('-levels', type=bool, default=False) # execute data labeling using all levels
parser.add_argument('-umls_fpath', type=Path, default= 'UMLS/english_subset/umls_preprocessed/umls_tui_pio3_.db')
parser.add_argument('-ds_fpath', type=Path, default='/mnt/nas2/data/systematicReview/ds_cto_dict' )
parser.add_argument('-abb_fpath', type=Path, default='/mnt/nas2/data/systematicReview/abbreviations' )
parser.add_argument('-indir', type=Path, default='/mnt/nas2/data/systematicReview' ) # directory with labeling function sources
parser.add_argument('-outdir', type=Path, default=f'/mnt/nas2/results/Results/systematicReview/distant_pico/training_ebm_candidate_generation/{candgen_version}' ) # directory path to store the weakly labeled candidates
parser.add_argument('-stop', type=bool, default=if_stopwords ) # False = Wont have stopword LF, True = Will have stopword LF
parser.add_argument('-write_cand', type=bool, default=False ) # Should write candidates? True = Yes - Write , False = No - Dont write
args = parser.parse_args()

try:

    ##############################################################################################################
    # Load training, validation and test datasets
    # Datasets used - EBM-PICO, Hilfiker physio
    ##############################################################################################################
    ebm_nlp = '/mnt/nas2/data/systematicReview/PICO_datasets/EBM_parsed'
    df_data, df_data_flatten = loadEBMPICO( ebm_nlp, args.outdir, candgen_version=candgen_version, write_to_file = False )

    #########################################################################################
    # Level 1 - UMLS LF's
    #########################################################################################
    if args.level1 == True or args.levels == True:

        # umls_tui_pio2 is used to generate UMLS2 candidate annotations: /systematicReview/distant_pico/candidate_generation/UMLS2
        # umls_v2.db is used to generate UMLS candidate annotations: /systematicReview/distant_pico/candidate_generation/UMLS
        
        umls_db = f'{args.indir}/{args.umls_fpath}'
        print('Retrieving UMLS ontology arm (Preprocessing applied)')
        umls_p  = loadUMLSdb(umls_db, entity='P')
        umls_i = loadUMLSdb(umls_db, entity='I')
        umls_o = loadUMLSdb(umls_db, entity='O')

        # for m in ['direct', 'fuzzy']: # fuzzy = fuzzy bigram match, direct = no fuzzy bigram match
        for m in ['direct', 'fuzzy']: # fuzzy = fuzzy bigram match, direct = no fuzzy bigram match
            outdir_umls = f'{args.outdir}/UMLS/{m}'
            # for entity, umls_d in zip(['P', 'I', 'O'], [ umls_p, umls_i, umls_o ]) :
            # for entity, umls_d in zip(['I', 'O'], [ umls_i, umls_o ]) :
            # for entity, umls_d in zip(['P'], [ umls_p ]):
            # for entity, umls_d in zip(['I'], [ umls_i ]):
            for entity, umls_d in zip(['O'], [ umls_o ]):
                label_umls_and_write(outdir_umls, umls_d, df_data, picos=entity, arg_options=args, write= args.write_cand )


    #########################################################################################
    # Level 2 - Non-UMLS LF's (non-UMLS Ontology labeling)
    #########################################################################################
    if args.level2 == True or args.levels == True:

        print('Retrieving non-UMLS Ontologies  (Preprocessing applied)')
        p_DO, p_DO_syn = loadOnt( f'{args.indir}/Ontologies/participant/DOID.csv', delim = ',', term_index = 1, term_syn_index = 2  )
        p_ctd, p_ctd_syn = loadOnt( f'{args.indir}/Ontologies/participant/CTD_diseases.tsv', delim = '\t', term_index = 0, term_syn_index = 7 )
        p_HPO, p_HPO_syn = loadOnt( f'{args.indir}/Ontologies/participant/HP.csv', delim = ',', term_index = 1, term_syn_index = 2  )
        i_ctd, i_ctd_syn = loadOnt( f'{args.indir}/Ontologies/intervention/CTD_chemicals.tsv', delim = '\t', term_index = 0, term_syn_index = 7 )
        i_chebi, i_chebi_syn = loadOnt( f'{args.indir}/Ontologies/intervention/CHEBI.csv', delim = ',', term_index = 1, term_syn_index = 2  )
        o_oae, o_oae_syn = loadOnt( f'{args.indir}/Ontologies/outcome/OAE.csv', delim=',', term_index=1, term_syn_index=2 )

        for m in ['fuzzy', 'direct']:
            outdir_non_umls = f'{args.outdir}/nonUMLS/{m}'
            for ontology, ont_name in zip([p_HPO, p_HPO_syn, p_DO, p_DO_syn, p_ctd, p_ctd_syn], ['HPO', 'HPO_syn', 'DO', 'DO_syn', 'CTD', 'CTD_syn'] ) :
               nonUMLS_p_labels = label_ont_and_write( outdir_non_umls, ontology, picos='P', df_data=df_data, write=args.write_cand, arg_options=args, ontology_name=ont_name)

            for ontology, ont_name in zip([i_ctd, i_ctd_syn, i_chebi, i_chebi_syn], ['CTD', 'CTD_syn', 'chebi', 'chebi_syn'] ) :
                nonUMLS_i_labels = label_ont_and_write( outdir_non_umls, ontology, picos='I', df_data=df_data, write=args.write_cand, arg_options=args, ontology_name=ont_name)

            for ontology, ont_name in zip([o_oae, o_oae_syn ], ['oae', 'oae_syn'] ) :
                nonUMLS_o_labels = label_ont_and_write( outdir_non_umls, ontology, picos='O', df_data=df_data, write=args.write_cand, arg_options=args, ontology_name=ont_name)

    #########################################################################################
    # Level 3 - Distant Supervision LF's
    #########################################################################################
    if args.level3 == True or args.levels == True:

        print('Retrieving distant supervision dictionaries')
        ds_participant = loadDS(args.ds_fpath, 'participant')
        ds_intervention = loadDS(args.ds_fpath, 'intervention')
        ds_intervention_syn = loadDS(args.ds_fpath, 'intervention_syn')
        ds_outcome = loadDS(args.ds_fpath, 'outcome')

        # for m in ['direct', 'fuzzy']:
        for m in ['fuzzy']:
            outdir_ds = f'{args.outdir}/ds/{m}'
            outdir_ds = f'/mnt/nas2/results/Results/systematicReview/order_free_matching/EBM_PICO_training_matches/{m}' # order-free matching
            for ontology, entity, ont_name in zip([ds_participant], ['P'], ['ds_participant'] ) :
                ds_p_labels = label_ont_and_write( outdir_ds, ontology, picos=entity, df_data=df_data, write=args.write_cand, arg_options=args, ontology_name=ont_name)

            for ontology, entity, ont_name in zip([ds_intervention, ds_intervention_syn], ['I', 'I'], ['ds_intervetion', 'ds_intervention_syn'] ) :
                ds_i_labels = label_ont_and_write( outdir_ds, ontology, picos=entity, df_data=df_data, write=args.write_cand, arg_options=args, ontology_name=ont_name)
        
            for ontology, entity, ont_name in zip([ds_outcome], ['O'], ['ds_outcome'] ) :
                ds_o_labels = label_ont_and_write( outdir_ds, ontology, picos=entity, df_data=df_data, write=args.write_cand, arg_options=args, ontology_name=ont_name)
    

    ##############################################################################################################
    # Level 4 - Rule based LF's (ReGeX, Heuristics and handcrafted dictionaries)
    ##############################################################################################################
    if args.level4 == True or args.levels == True:

        print('Retrieving ReGeX patterns')
        p_sampsize = loadPattern( 'samplesize' ) # Generic pattern 
        p_sampsize2 = loadPattern( 'samplesize2' ) # Sample size in ReGeX expression (n=XXXX)
        p_sampsize3 = loadPattern( 'samplesize3' )  # Digits expressed as words 0_999
        p_sampsize4 = loadPattern( 'samplesize4' )  # Digits expressed as words one_to_999
        p_sampsize5 = loadPattern( 'samplesize5' )  # Digits expressed as words one_to_999_999

        p_age = loadPattern( 'age0' )
        p_agerange = loadPattern( 'age1' )
        p_agemax = loadPattern( 'age2' )
        p_agemaxmin = loadPattern( 'age3' )
        p_meanage = loadPattern( 'meanage' )

        i_control = loadPattern( 'control_i' )

        s_study_type = loadPattern( 'studytype' )
        s_study_type_basic = loadPattern( 'studytype_basic' )
        s_study_type_basicplus = loadPattern( 'studytype_basic+' )
        s_study_type_proc = loadPattern( 'studytype_procedure' )
        s_study_type_s = loadPattern( 'studytypes_var' )

        # # ReGeX Labeling Function
        for reg_lf_i, entity, reg_lf_name in zip([p_sampsize, p_sampsize2, p_sampsize3, p_sampsize4, p_sampsize5, p_age, p_agerange, p_agemax, p_agemaxmin, p_meanage, i_control, s_study_type, s_study_type_basic, s_study_type_basicplus, s_study_type_s], ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'I', 'S', 'S', 'S', 'S'], ['regex_sampsize', 'regex_sampsize2', 'regex_sampsize3', 'regex_sampsize4', 'regex_sampsize5', 'regex_age', 'regex_agerange', 'regex_agemax', 'regex_agemaxmin', 'regex_meanage', 'regex_comparator', 'regex_stdtype', 'regex_stdtype_basic', 'regex_stdtype_types'] ) : 
        # for reg_lf_i, entity, reg_lf_name in zip([ p_sampsize3 ], ['P'], ['samplesize3'] ) : 
            outdir_reg = f'{args.outdir}/heuristics/direct'
            label_regex_and_write( outdir_reg, [reg_lf_i], picos=entity, df_data=df_data, write=args.write_cand, arg_options=args, lf_name=reg_lf_name )

        # # Heutistic Labeling Functions
        outdir_heurPattern = f'{args.outdir}/heuristics/direct'

        filename = 'lf_' + str('i_posreg') + '.tsv'
        label_heur_and_write( outdir_heurPattern, picos='I', df_data=df_data, write=args.write_cand, arg_options=args, lf_name=str('i_posreg'))

        filename = 'lf_' + str('pa_regex_heur') + '.tsv'
        label_heur_and_write( outdir_heurPattern, picos='P', df_data=df_data, write=args.write_cand, arg_options=args, lf_name=str(filename).replace('.tsv', ''))

        filename = 'lf_' + str('ps_heurPattern_labels') + '.tsv'
        label_heur_and_write( outdir_heurPattern, picos='P', df_data=df_data, write=args.write_cand, arg_options=args, lf_name=str(filename).replace('.tsv', ''))

       
        print('Retrieving hand-crafted dictionaries')
        p_genders = loadDict(f'{args.indir}/Ontologies/participant/gender_sexuality.txt')
        i_comparator = loadDict(f'{args.indir}/Ontologies/intervention/comparator_dict.txt')
        o_endpoints = loadDict(f'{args.indir}/Ontologies/outcome/endpoints_dict.txt')
        s_dictionary = loadDict(f'{args.indir}/Ontologies/study_type/rct.txt')

        print('Retrieving abbreviations dictionaries')  
        p_abb = loadAbbreviations(f'{args.indir}/Ontologies/participant/diseaseabbreviations.tsv')

        # Dictionary Labeling Function
        for m in ['fuzzy', 'direct']:
            for ontology, entity, ont_name in zip([p_genders, i_comparator, o_endpoints, s_dictionary], ['P', 'I', 'O', 'S'], ['dict_gender', 'dict_comparator', 'dict_o_terms', 'dict_s_type'] ) : 
                outdir_dict = f'{args.outdir}/dictionary/{m}'
                dict_labels = label_ont_and_write( outdir_dict, ontology, picos=entity, df_data=df_data, write=args.write_cand, arg_options=args, ontology_name=ont_name)

    #########################################################################################
    # TODO  Level 5 - Abbreviations Labeling function
    #########################################################################################
    if args.level5 == True or args.levels == True:

        if extract_abbs == True:

            umls_db = f'{args.indir}/{args.umls_fpath}'
            print('Retrieving UMLS ontology arm (Preprocessing applied)')
            umls_p  = loadUMLSdb(umls_db, entity='P')
            positive_p, negative_p = loadAbbreviationDicts(umls_p)

            umls_i = loadUMLSdb(umls_db, entity='I')
            positive_i, negative_i = loadAbbreviationDicts(umls_i)

            umls_o = loadUMLSdb(umls_db, entity='O')
            positive_o, negative_o = loadAbbreviationDicts(umls_o)

            ds_participant = loadDS(args.ds_fpath, 'participant')
            ds_intervention = loadDS(args.ds_fpath, 'intervention')
            ds_intervention_syn = loadDS(args.ds_fpath, 'intervention_syn')
            ds_outcome = loadDS(args.ds_fpath, 'outcome')

            p_DO, p_DO_syn = loadOnt( f'{args.indir}/Ontologies/participant/DOID.csv', delim = ',', term_index = 1, term_syn_index = 2  )
            p_ctd, p_ctd_syn = loadOnt( f'{args.indir}/Ontologies/participant/CTD_diseases.tsv', delim = '\t', term_index = 0, term_syn_index = 7 )
            p_HPO, p_HPO_syn = loadOnt( f'{args.indir}/Ontologies/participant/HP.csv', delim = ',', term_index = 1, term_syn_index = 2  )
            p_abb = loadAbbreviations(f'{args.indir}/Ontologies/participant/diseaseabbreviations.tsv')
            i_ctd, i_ctd_syn = loadOnt( f'{args.indir}/Ontologies/intervention/CTD_chemicals.tsv', delim = '\t', term_index = 0, term_syn_index = 7 )
            i_chebi, i_chebi_syn = loadOnt( f'{args.indir}/Ontologies/intervention/CHEBI.csv', delim = ',', term_index = 1, term_syn_index = 2  )
            o_oae, o_oae_syn = loadOnt( f'{args.indir}/Ontologies/outcome/OAE.csv', delim=',', term_index=1, term_syn_index=2 )

            for i in [ds_participant, p_DO, p_DO_syn, p_ctd, p_ctd_syn, p_HPO, p_HPO_syn, p_abb]:
                positive_p.update(set( i ))

            for i in [ds_intervention, ds_intervention_syn, i_ctd, i_ctd_syn, i_chebi, i_chebi_syn]:
                positive_i.update(set( i ))

            for i in [ds_outcome, o_oae, o_oae_syn]:
                positive_o.update(set( i ))

            for ontology, entity in zip([ (positive_p, negative_p) , (positive_i, negative_i), (positive_o, negative_o) ], ['P', 'I', 'O'] ) : 
                abbs = AbbreviationFetcher( df_data, ontology[0], ontology[1], entity)


        # Load abbreviations into a dictionary
        def get_abbs(entity):

            abb_d = {}

            entity_full = { 'P': 'participant', 'I': 'intervention', 'O': 'outcome' }

            with open( f'{args.abb_fpath}/{entity_full[entity]}/abb_sources.json', 'r' ) as af:
                for l in af:
                    data_i = json.loads(l)
                    abb_d.update( data_i )

            return abb_d

        abb_p = get_abbs('P')
        abb_i = get_abbs('I')
        abb_o = get_abbs('O')

        for m in ['direct']:
            for abb, entity, ont_name in zip([abb_p, abb_i, abb_o], ['P', 'I', 'O'], ['abb_p', 'abb_i', 'abb_o'] ) : 
                outdir_dict = f'{args.outdir}/heuristics/{m}'
                label_abb_and_write(outdir_dict, abb, entity, df_data=df_data, write=args.write_cand, arg_options=args, lf_name=ont_name)

        print('Retrieving abbreviations dictionaries')  
        p_abb = loadAbbreviations(f'{args.indir}/Ontologies/participant/diseaseabbreviations.tsv')
        s_abb = loadAbbreviations(f'{args.indir}/Ontologies/study_type/studytype_abbreviations.tsv')
        # Dictionary Labeling Function and Abbreviation dictionary Labeling function
        for m in ['direct']:
            for ontology, entity, ont_name in zip([p_abb, s_abb], ['P', 'S'], ['dict_p_abb', 'dict_s_abb'] ) : 
                outdir_dict = f'{args.outdir}/heuristics/{m}'
                dict_labels = label_ont_and_write( outdir_dict, ontology, picos=entity, df_data=df_data, write=args.write_cand, arg_options=args, ontology_name=ont_name)


    #########################################################################################
    # TODO  Level 6 - External Model Labeling function
    #########################################################################################
    if args.level6 == True or args.levels == True:
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
