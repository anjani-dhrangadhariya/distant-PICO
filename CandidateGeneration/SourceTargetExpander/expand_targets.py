#!/usr/bin/env python

def ExpandTargetsDoc(a):
    '''The module expands Individual annotation targets.'''
    return a**a

print( ExpandTargetsDoc.__doc__ )


# imports - general
import time

# imports - linguistic
import re
import spacy
from scispacy.abbreviation import AbbreviationDetector

# 
from SourceTargetExpander.SourceExpander.expansion_utils import *


nlp = spacy.load("en_core_sci_sm")
# Add the abbreviation detector to spacy pipeline
nlp.add_pipe("abbreviation_detector")

def expandTargets(json_object, targets):

    expanded_targets = dict()

    for key, value in targets.items():
        # print( value )
        newline_removed = value.replace("\n", " ").replace("\r", " ")
        trailingspaces_removed = removeSpaceTrailsString(newline_removed)
        # print( trailingspaces_removed )
        expanded_targets[key] = trailingspaces_removed


    print( expanded_targets )