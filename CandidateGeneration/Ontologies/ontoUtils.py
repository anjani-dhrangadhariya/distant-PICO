import random
import pandas as pd
import spacy

#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words
import string

# lower-casing without altering the Abbreviations
from Ontologies.transforms import SmartLowercase
lower_caser =  SmartLowercase()

additional_stopwords = ['of']
stopwords.update(additional_stopwords)
def filterSAB():
    return ['SNOMEDCT_VET', 'NCI_ZFin', 'NCI_ICDC', 'NCI_JAX'] # vet UMLS dictionaries




'''
Description:
    This function could be used for preprocessing ontology terms. Preprocessing includes 1) Remove stopwords 2) Remove numbers 3) Remove punctuations

Args:
    term (str): String variable containing the ontology term

Returns:
    preprocessed term (str): String variable containing the preprocessed ontology term
'''
def preprocessOntology(term):

    # remove stopwords
    lst = [ token for token in term.split() if token.lower() not in stopwords ]
    lst = ' '.join(lst)

    # remove numbers
    numRemove = ''.join([i for i in lst if not i.isdigit()])

    # remove punctuation
    punctRemove = numRemove.translate(str.maketrans(' ', ' ', string.punctuation))

    return punctRemove

def allowedTermLength(term):
    return True if len(term.split()) > 1 else False

def countTerm(umls):

    flagger = 0
    for k,v in umls.items():
        if len(v) > 500:
            flagger = flagger + 1
    return flagger

def removeNonHuman(umls_d):

    # Load the non-human ontology filter
    non_human_umls = filterSAB()

    for i in non_human_umls:
        umls_d.pop(i, None)

    return umls_d


def termCountThreshold(umls_d):

    list_k = [k for k, v in umls_d.items() if len(v) < 500]
    [ umls_d.pop(i, None) for i in list_k ]

    return umls_d

def removeTerms( umls_d, char_threshold ):

    filtered_onto = dict()
    for k,v in umls_d.items():
        temp_v = [ v_i for v_i in v if len(v_i[0]) > char_threshold  ]
        filtered_onto[k] = temp_v

    return filtered_onto

def smart_lower_Case(umls_d):

    lowercased_umls_d = dict()
    for k,v in umls_d.items():
        
        v_new = []

        for v_i in v:
            v_i_0_new = lower_caser(v_i[0])
            v_i_new = ( v_i_0_new, v_i[1] )
            v_new.append(v_new)

        lowercased_umls_d[k] = v_new

    return lowercased_umls_d