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
    The funtion expands on the Intervention/Comparator terms of the clinical trial study using heuristics and adds POS tags  and abbreviation information

Args:
    dictionary value (string): free-text describing study interventions
        fetch_pos (bool): True (default)
        fetch_abb (bool): True (default)

Returns:
    dictionary: returns a dictionary of expanded Intervention/Comparator terms along with POS tags and abbreviation information
'''
def expandIntervention(intervention_source, fetch_pos = True, fetch_abb = True):

    expanded_intervention = dict()

    for key, value in intervention_source.items():
        if 'arm' not in key:
            
            if '&' in value and 'vs' not in value and ',' not in value  and ':' not in value and '(' not in value: # ampersand              
                values = value.split('&')
                values.append( value )
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, values, key)
                else:
                    expanded_intervention[key] = values

            elif '&' not in value and 'vs' in value and ',' not in value  and ':' not in value and '/' not in value and '(' not in value: # versus
                values = value.split('vs')
                values.append( value )
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, values, key)
                else:
                    expanded_intervention[key] = values

            elif '&' not in value and 'vs' not in value and ',' not in value  and ':' in value and '/' not in value and '(' not in value: # semi-colon
                values = value.split(':')
                values.append( value )
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, values, key)
                else:
                    expanded_intervention[key] = values

            elif '&' not in value and 'vs' not in value and ',' in value  and ':' not in value and '/' not in value and '(' not in value: # comma
                values = value.split(',')
                values.append( value )
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, values, key)
                else:
                    expanded_intervention[key] = values

            elif '&' not in value and 'vs' not in value and ',' not in value  and ':' not in value and '/' in value and '(' not in value: # forward slash
                values = value.split('/')
                values.append( value )
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, values, key)
                else:
                    expanded_intervention[key] = values

            else:
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, [value], key)
                else:
                    expanded_intervention[key] = values

        else:
            expanded_intervention[key] = value

    return expanded_intervention