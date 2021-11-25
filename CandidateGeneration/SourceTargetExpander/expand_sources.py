#!/usr/bin/env python

def ExpandSourcesDoc(a):
    '''The module expands Individual annotation sources.'''
    return a**a

print( ExpandSourcesDoc.__doc__ )

# imports - general
import time

# imports - linguistic
import re
import spacy
from scispacy.abbreviation import AbbreviationDetector


nlp = spacy.load("en_core_sci_sm")
# Add the abbreviation detector to spacy pipeline
nlp.add_pipe("abbreviation_detector")

from SourceTargetExpander.SourceExpander.expand_participants import *
from SourceTargetExpander.SourceExpander.expand_interventions import *
from SourceTargetExpander.SourceExpander.expand_outcomes import *
from SourceTargetExpander.SourceExpander.expand_stdtype import *

def expandSources_i(json_object, sources):

    organized_sources = dict()

    if 'MaximumAge' in sources['p_age']:
        # del my_dict['key']
        del sources['p_age']['MaximumAge']

    if 'MimimumAge' in sources['p_age']:
        # del my_dict['key']
        del sources['p_age']['MimimumAge']

    if 'StdAge' in sources['p_age']:
        sources['p_age'] = sources['p_age']['StdAge']

    return sources

def expandSources_ii(json_object, sources):

    expanded_sources = dict()

    # -------------------------------------------------------------------------------------
    # Participant subclasses 
    # -------------------------------------------------------------------------------------
    if 'p_sample_size' in sources:
        expanded_sampsize = expandSampleSize([sources['p_sample_size']])  # Expanded sample size is a ReGeX
        expanded_sources['ep_sample_size'] = expanded_sampsize

    if 'p_gender' in sources:
        expanded_gender = expandGender(sources['p_gender'])   # Expanded gender is a dictionary
        expanded_sources['ep_gender'] = expanded_gender

    if 'p_age' in sources:
        expanded_age = expandAge(sources['p_age'])
        expanded_sources['ep_age'] = expanded_age

    if 'p_condition' in sources:
        expanded_condition = expandCondition(sources['p_condition'], fetch_pos=False, fetch_abb=True)
        expanded_sources['ep_condition'] = expanded_condition

    # -------------------------------------------------------------------------------------
    # I/C Interventions Comparators 
    # -------------------------------------------------------------------------------------
    if 'i_name' in sources:
        expanded_intervention = expandIntervention(sources['i_name'], fetch_pos=True, fetch_abb=True)
        expanded_sources['ei_name'] = expanded_intervention

    # -------------------------------------------------------------------------------------
    # Outcomes
    # -------------------------------------------------------------------------------------
    if 'o_name' in sources:
        expanded_prim_outcomes = expandOutcomes(sources['o_name'])
        expanded_sources['eo_name'] = expanded_prim_outcomes

    # -------------------------------------------------------------------------------------
    # Study type
    # -------------------------------------------------------------------------------------
    if 's_type' in sources:
        expanded_studytype = expandStudyType(sources['s_type'])
        expanded_sources['es_type'] = expanded_studytype

    return expanded_sources