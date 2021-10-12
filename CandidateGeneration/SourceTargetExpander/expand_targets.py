#!/usr/bin/env python

def ExpandTargetsDoc(a):
    '''The module expands Individual annotation targets.'''
    return a**a

print( ExpandTargetsDoc.__doc__ )


# imports - general
import enum
import time

# imports - linguistic
import re
import spacy
from scispacy.abbreviation import AbbreviationDetector
from spacy.lang.en import English

nlp = English()
nlp.add_pipe("sentencizer")

# 
from SourceTargetExpander.SourceExpander.expansion_utils import *

'''
TODO
Description:
    The function preprocesses and expands the targets

Args:
    String: free-text string
        

Returns:
    Dictionary (dict): expanded targets
'''
def expandTargets(json_object, targets):

    expanded_targets = dict()

    for key, value in targets.items():

        newline_removed = value.replace("\n", " ").replace("\r", " ")
        trailingspaces_removed = removeSpaceTrailsString(newline_removed)

        sep = 'Exclusion Criteria'
        if key == 'EligibilityCriteria':
            if sep in trailingspaces_removed:
                trailingspaces_removed = trailingspaces_removed.split(sep, 1)[0]

        # Convert long targets to sentences
        target_sentences = nlp(trailingspaces_removed)

        sent_parsed = dict()
        
        for ith, eachSent in enumerate(target_sentences.sents):

            # Get POS-tags for the target sub-string
            possed_targets = getPOStags( str(eachSent) )
            nested_key = 'sentence_' + str(ith)
            sent_parsed[nested_key] = possed_targets

        if key not in expanded_targets:
            expanded_targets[key] = sent_parsed

    return expanded_targets