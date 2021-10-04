#!/usr/bin/env python

def ExpandSourcesDoc(a):
    '''The module generates distant annotations for PICO entities using a cominbation of distant supervision and dynamic programming'''
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

def expandSources(json_object, sources):

    expanded_sources = dict()

    # P
    expanded_gender = expandGender(sources['p_gender'])
    expanded_age = expandAge(sources['p_age'])
    # P - Condition needs abbreviation detection
    # print( sources['p_condition'] )
    # P - Sample size does not need expansion
    # print( sources['p_sample_size'] )

    # I/C
    expanded_intervention = expandIntervention(sources['i_name'], fetch_pos=True, fetch_abb=True)
    # I - synonyms do not require any expansion
    #    print( sources['i_synonym'] )

    # O
    # Outcomes do not require other expansion except POS tagging
    expanded_prim_outcomes = expandOutcomes(sources['o_primary'])
    expanded_second_outcomes = expandOutcomes(sources['o_secondary'])

    # S
    expanded_studytype = expandStudyType(sources['s_type'])

    expanded_sources['ep_gender'] = expanded_gender
    expanded_sources['ep_age'] = expanded_age

    expanded_sources['ei_name'] = expanded_intervention

    expanded_sources['eo_primary'] = expanded_prim_outcomes
    expanded_sources['eo_secondary'] = expanded_second_outcomes

    expanded_sources['es_type'] = expanded_studytype

    return expanded_sources