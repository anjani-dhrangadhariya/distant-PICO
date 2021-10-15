# imports - general
import time

from SourceTargetExpander.SourceExpander.expansion_utils import *

'''
Description:
    The funtion expands the Intervention/Comparator terms extracted from the clinical trial study using 'if/else' heuristics and adds POS tags and abbreviations using scispacy

Args:
    dictionary (dict): dictionary storing "intervention" and "arms groups" terms
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

            elif '&' not in value and 'vs' in value and ',' not in value  and ':' not in value and '/' not in value and '(' not in value: # versus
                values = value.split('vs')
                values.append( value )

            elif '&' not in value and 'vs' not in value and ',' not in value  and ':' in value and '/' not in value and '(' not in value: # semi-colon
                values = value.split(':')
                values.append( value )

            elif '&' not in value and 'vs' not in value and ',' in value  and ':' not in value and '/' not in value and '(' not in value: # comma
                values = value.split(',')
                values.append( value )

            elif '&' not in value and 'vs' not in value and ',' not in value  and ':' not in value and '/' in value and '(' not in value: # forward slash
                values = value.split('/')
                values.append( value )

            elif '&' not in value and 'vs' not in value and ',' not in value  and ':' not in value and '/' not in value and '(' in value: # abbreviations
                abbreviations = fetchAcronyms(value)
                if abbreviations is not None:
                    values = abbreviations
                else:
                    values = [value]

            else:
                values = [value]

            # After retrieving all the abbreviations, add the POS tags
            for i, eachValue in enumerate(values):
                key1 = key + '_' + str(i)
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, [eachValue], key1)

        # else: # XXX: Arms group intervention terms here! Won't be used for experimentation or candidate generation
        #     expanded_intervention[key] = value

    return expanded_intervention