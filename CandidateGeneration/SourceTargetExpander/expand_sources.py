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

def expand_o(json_object, sources):

    
    # if 'o_name' in sources:
    #     expanded_prim_outcomes = expandOutcomes(sources['o_name'])
    #     expanded_sources['eo_name'] = expanded_prim_outcomes

    # # -------------------------------------------------------------------------------------
    # # Study type
    # # -------------------------------------------------------------------------------------
    # if 's_type' in sources:
    #     expanded_studytype = expandStudyType(sources['s_type'])
    #     expanded_sources['es_type'] = expanded_studytype

    return expanded_intervention