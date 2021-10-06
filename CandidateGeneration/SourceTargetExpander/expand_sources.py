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

def expandSources(json_object, sources):

    expanded_sources = dict()

    # -------------------------------------------------------------------------------------
    # P
    # -------------------------------------------------------------------------------------
    expanded_gender = expandGender(sources['p_gender'])
    expanded_age = expandAge(sources['p_age'])
    expanded_condition = expandCondition(sources['p_condition'])
    print( expanded_condition )

    expanded_sources['ep_gender'] = expanded_gender
    expanded_sources['ep_age'] = expanded_age
    expanded_sources['ep_condition'] = expanded_condition
    expanded_sources['ep_sample_size'] = sources['p_sample_size']

    # -------------------------------------------------------------------------------------
    # I/C
    # -------------------------------------------------------------------------------------
    expanded_intervention_ = expandIntervention(sources['i_name'], fetch_pos=True, fetch_abb=True)

    if bool(sources['i_synonym']) == True:
        expanded_intervention = {**expanded_intervention_, **sources['i_synonym']}
    elif bool(sources['i_synonym']) == False:
        expanded_intervention = expanded_intervention_

    expanded_sources['ei_name'] = expanded_intervention

    # -------------------------------------------------------------------------------------
    # O
    # -------------------------------------------------------------------------------------
    expanded_prim_outcomes = expandOutcomes(sources['o_primary'])
    expanded_second_outcomes = expandOutcomes(sources['o_secondary'])

    expanded_sources['eo_primary'] = expanded_prim_outcomes
    expanded_sources['eo_secondary'] = expanded_second_outcomes

    # -------------------------------------------------------------------------------------
    # S
    # -------------------------------------------------------------------------------------
    expanded_studytype = expandStudyType(sources['s_type'])

    expanded_sources['es_type'] = expanded_studytype

    return expanded_sources