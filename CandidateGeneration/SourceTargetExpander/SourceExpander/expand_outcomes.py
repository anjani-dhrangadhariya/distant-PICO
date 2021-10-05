# imports - general
import time

# imports - linguistic
import re
import spacy
from scispacy.abbreviation import AbbreviationDetector

from SourceTargetExpander.SourceExpander.expansion_utils import *

nlp = spacy.load("en_core_sci_sm")
# Add the abbreviation detector to spacy pipeline
nlp.add_pipe("abbreviation_detector")

'''
Description:
    The funtion expands on the outcomes of the clinical trial study using heuristics and adds POS tags  and abbreviation information

Args:
    dictionary value (string): free-text describing study interventions
        fetch_pos (bool): True (default)
        fetch_abb (bool): True (default)

Returns:
    dictionary: returns a dictionary of expanded outcome terms along with POS tags and abbreviation information
'''
def expandOutcomes(outcome_source, fetch_pos = True, fetch_abb = True):

    expanded_outcome = dict()

    for key, value in outcome_source.items():
        if fetch_pos == True:
            expanded_outcome = appendPOSSED(expanded_outcome, [value], key)
        else:
            expanded_outcome[key] = value

    return expanded_outcome