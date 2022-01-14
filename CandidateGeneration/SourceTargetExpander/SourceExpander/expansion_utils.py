# imports - general
import time

# imports - linguistic
import re
import spacy
from scispacy.abbreviation import AbbreviationDetector

nlp = spacy.load("en_core_sci_sm")
# Add the abbreviation detector to spacy pipeline
nlp.add_pipe("abbreviation_detector")

'''
Description:
    The function removes trailing spaces from a string and replaces them with a single string

Args:
    string (str): free-text string

Returns:
    string (str): clean string 
'''
def removeSpaceTrailsString(s):
    return " ".join(s.split())

'''
Description:
    The function extracts acronyms using Scispacy abbreviation detector

Args:
    string (str): free-text string

Returns:
    List (list): tuple with detected abbreviation and the long form of the abbreviation
'''
def fetchAcronyms(value):

    doc = nlp(value)

    altered_tok = [tok.text for tok in doc]

    abbreviations = []

    for abrv in doc._.abbreviations:
        altered_tok[abrv.start] = str(abrv._.long_form)

        # print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form} \t  {value}")
        # print( " ".join(altered_tok) )

        abbreviations.append( str(abrv) )
        abbreviations.append( str(abrv._.long_form) )
        # original = str(abrv._.long_form) + ' (' + str(abrv) + ')'
        # abbreviations.append( original  )

    if abbreviations:
        return abbreviations

'''
Description:
    The function fetches POS tags for the input string

Args:
    String: free-text string

Returns:
    Dictionary (dict): returns a dictionary containing free-text string with its tokenized string, token lemma, POS tags for the tokens, finer POS tags for the token
'''
def getPOStags(value):

    doc = nlp(value)

    pos_dict = dict()

    tokens = [ token.text for token in doc ]
    lemma = [ token.lemma_ for token in doc ]
    pos = [ token.pos_ for token in doc ]
    pos_fine = [ token.tag_ for token in doc ]
    depth = [ token.dep_ for token in doc ]
    shape = [ token.shape_ for token in doc ]
    is_alpha = [ token.is_alpha for token in doc ]
    is_stop = [ token.is_stop for token in doc ]

    pos_dict['text'] = value
    pos_dict['tokens'] = tokens
    pos_dict['lemma'] = lemma
    pos_dict['pos'] = pos
    pos_dict['pos_fine'] = pos_fine   

    return pos_dict

'''
TODO
Description:
    

Args:
    Dictionary (dict): free-text string
        value (string):
        key (string):

Returns:
    Dictionary (dict):
'''
def appendPOSSED(expanded_dictionary, values, key):

    for value_i in values:
        possed_value = getPOStags(value_i)
        if key not in expanded_dictionary:
            expanded_dictionary[key] = [possed_value]
        else:
            expanded_dictionary[key].append( possed_value )

    return expanded_dictionary