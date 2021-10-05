#!/usr/bin/env python

def ExpandTargetsDoc(a):
    '''The module expands Individual annotation targets.'''
    return a**a

print( ExpandTargetsDoc.__doc__ )


# imports - general
import time

# imports - linguistic
import re

# 
from SourceTargetExpander.SourceExpander.expansion_utils import *

def expandTargets(json_object, targets):

    expanded_targets = dict()

    for key, value in targets.items():
        newline_removed = value.replace("\n", " ").replace("\r", " ")
        trailingspaces_removed = removeSpaceTrailsString(newline_removed)

        # Get POS-tags for the target string
        possed_targets = getPOStags(trailingspaces_removed)
        expanded_targets[key] = possed_targets

    return expanded_targets