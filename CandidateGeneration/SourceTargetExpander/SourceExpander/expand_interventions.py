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

    values = []

    for key, value in intervention_source.items():

        if 'arm' not in key:
            
            if '&' in value and 'vs' not in value and ',' not in value  and ':' not in value and '(' not in value and '[' not in value: # ampersand              
                value_ = [x.strip() for x in value.split('&')]
                values.extend( value_ )

            elif '&' not in value and 'vs' in value and ',' not in value  and ':' not in value and '/' not in value and '(' not in value and '[' not in value: # versus
                value_ = [x.strip() for x in value.split('vs')]
                values.extend( value_ )

            elif '&' not in value and 'vs' not in value and ',' not in value  and ';' in value and '/' not in value and '(' not in value and '[' not in value: # semi-colon
                value_ = [x.strip() for x in value.split(';')]
                values.extend( value_ )

            elif '&' not in value and 'vs' not in value and ',' in value  and ':' not in value and '/' not in value and '(' not in value and '[' not in value: # comma
                value_ = [x.strip() for x in value.split(',')]
                values.extend( value_ )

            elif '&' not in value and 'vs' not in value and ',' not in value  and ':' not in value and '/' in value and '(' not in value and '[' not in value: # forward slash
                value_ = [x.strip() for x in value.split('/')]
                values.extend( value_ )

            elif '&' not in value and 'vs' not in value and ',' not in value  and ':' not in value and '/' not in value and ('(' in value or '[' in value): # abbreviations
                abbreviations = fetchAcronyms(value)
                if abbreviations is not None:
                    values.extend( abbreviations )
                else:
                    values.extend([value])

            else:
                values.extend([value])

    # After retrieving all the abbreviations, add the POS tags
    for i, eachValue in enumerate( list(set(values)) ):

        key1 = key + '_' + str(i)
        if fetch_pos == True:
            possed = getPOStags( eachValue )
            expanded_intervention[key1] = possed

    # else: # XXX: Arms group intervention terms here! Won't be used for experimentation or candidate generation
    #     expanded_intervention[key] = value

    return expanded_intervention