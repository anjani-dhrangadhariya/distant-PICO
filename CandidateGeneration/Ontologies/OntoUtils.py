import random
import pandas as pd
import spacy

#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words
import string

additional_stopwords = ['of']
stopwords.update(additional_stopwords)
def filterSAB():
    return ['SNOMEDCT_VET', 'NCI_ZFin', 'NCI_ICDC', 'NCI_JAX']

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

def rankSAB(umls_d):

    keys = [k for k in umls_d.keys()] 

    shuffle(keys)

    umls_d_new = dict()

    for k in keys:
        umls_d_new[k] = umls_d[k]

    return umls_d_new

def list2Nested(l, nested_length):
    return [l[i:i+nested_length] for i in range(0, len(l), nested_length)]

def partitionRankedSAB(umls_d):

    keys = list(umls_d.keys())

    partitioned_lfs = [ ]

    for i in range( 0, len(keys) ):

        if i == 0 or i == len(keys):
            if i == 0:
                partitioned_lfs.append( keys )
            if i ==len(keys):
                temp3 = list2Nested(keys, 1)
                partitioned_lfs.append( temp3 )
        else:
            temp1, temp2 = keys[:i] , keys[i:]
            temp3 = list2Nested( keys[:i], 1)
            temp3.append( keys[i:] )
            partitioned_lfs.append( temp3 )

    return partitioned_lfs