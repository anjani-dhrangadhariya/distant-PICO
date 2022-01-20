from LabelingFunctions import ontologyLF
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize, word_tokenize
from nltk import ngrams
import re
import sys
import os
import pandas as pd


pico2labelMap = dict()
pico2labelMap = { 'P' : 1, 'I' : 1, 'O' : 1, '-P' : 0, '-I' : 0, '-O' : 0, 'IO' : 0, 'OI' : 0, 'PO' : 0, 'OP' : 0, 'IP': 0, 'PI': 0, '-IO' : 0, '-OI' : 0, '-PO' : 0, '-OP' : 0, '-IP': 0, '-PI': 0, '-PIO' : 0, 'PIO' : 0 }

def flatten(t):
    return [item for sublist in t for item in sublist]

def expandTerm( term , max_ngram, fuzzy_match):
    
    expandedTerm = []
    termVariations = []

    if len( term.split() ) > max_ngram:
        fivegrams = ngrams(term.lower().split(), max_ngram)
        expandedTerm.extend( [' '.join(x)  for x in list(fivegrams)] )
    else:
        expandedTerm.extend( [term.lower()] )

    if fuzzy_match == True:
        bigrams = ngrams(term.lower().split(), 2)
        expandedTerm.extend( [' '.join(x)  for x in list(bigrams)] )

    for eT in expandedTerm:
        termVariations.extend( [eT.lower().rstrip('s'), eT.lower() + 's'] )

    return termVariations

'''
def char_to_word_index(ci, sequence):
    """
    Given a character-level index (offset),
    return the index of the **word this char is in**
    """
    i = None
    for i, co in enumerate(sequence):
        if ci == co:
            return i
        elif ci < co:
            return i - 1
    return i
'''

def char_to_word_index(ci, sequence):
    """
    Given a character-level index (offset),
    return the index of the **word this char is in**
    """
    i = None
    if ci in sequence:
        i = sequence[ci]

    return i

def get_word_index_span(char_offsets, sequence):
    char_start, char_end = char_offsets
    return (char_to_word_index(char_start, sequence),
            char_to_word_index(char_end, sequence))


def spansToLabels(matches, labels, terms, start_spans, generated_labels, text_tokenized):

    for m, t, l in zip(matches, terms, labels):
        
        for m_i in m:

            if len(m_i.group()) > 2:

                start, end = get_word_index_span(
                    (m_i.span()[0], m_i.span()[1] - 1), start_spans
                )

                if end and start:
                    match_temp = ' '.join( [text_tokenized[x]  for x in range( start, end+1 )] )
                    for x in range( start, end+1 ):
                        if isinstance( t , re._pattern_type ):
                            if len( match_temp.strip() ) == len(m_i.group().strip()):
                                # print( match_temp.strip() , ' ----- ',  len(m_i.group().strip()) )
                                generated_labels[x] = l
                        else:
                            if len( match_temp.strip() ) == len(t.strip()):
                                #print( match_temp.strip() , ' ----- ',  t.strip())
                                generated_labels[x] = l
                else:
                    # pass
                    print(start , ' : ', end , ' - ', t)

    return generated_labels


def heurspansToLabels(matches, spans, labels, start_spans, generated_labels, text_tokenized):

    for m, s, l in zip(matches, spans, labels):

        joined_m = ' '.join(m)
        if len( s ) == 1:
            joined_s = s[0]
        else:
            joined_s = ( s[0][0] , s[-1][-1] )

        start, end = get_word_index_span(
            (s[0][0], s[-1][-1] - 1), start_spans
        )
        if end and start:

            match_temp = ' '.join( [text_tokenized[x]  for x in range( start, end+1 )] )
            for x in range( start, end+1 ):
                if len( match_temp.strip() ) == len(joined_m.strip()):
                    #print( match_temp.strip() , ' ----- ',  joined_m.strip())
                    generated_labels[x] = l

    return generated_labels

def heurspansToLabels2(matches, labels, start_spans, generated_labels, text_tokenized):
    
    for m, l in zip(matches, labels):

        if len(m.group()) > 2:

            start, end = get_word_index_span(
                (m.span()[0], m.span()[1] - 1), start_spans
            )

            match_temp = ' '.join( [text_tokenized[x]  for x in range( start, end+1 )] )
            for x in range( start, end+1 ):
                if len( match_temp.strip() ) == len(m.group().strip()):
                    # print( match_temp.strip() , ' ----- ',  joined_m.strip())
                    generated_labels[x] = l
                # else:
                    # print( match_temp.strip() , ' ----- ',  m.group().strip())

    return generated_labels


def pico2label(l):

    l = [ pico2labelMap[ l_i ] if l_i != -1 else l_i for l_i in l ]

    return l

def write2File(labels_, tokens_, number_lfs, picos):
    
    indir = '/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation'
    
    df = pd.DataFrame( tokens_, columns=['tokens'] )
    
    for i in number_lfs:
        column_name = 'lf_' + str(i+1)
        df = df.merge( pd.DataFrame( labels_[i], columns=[column_name] ) , left_index=True, right_index=True )
    
    filename = 'lf_' + str(len(number_lfs)) + '.csv'
    with open(f'{indir}/{picos}/{filename}', 'w+') as wf:
        df.to_csv(f'{indir}/{picos}/{filename}', sep='\t')

def label_lf_partitions( partitioned_uml, umls_d, picos, text, token_flatten, spans, start_spans):

    accumulated_labels = []

    for i, lf in enumerate(partitioned_uml):

        print( 'Number of labeling functions: ', len(lf) )
        accumulated_lf = []
        number_lfs = []

        for number in range(0, len(lf)):

            sab_for_lf = lf[ number ]
            terms = flatten([ umls_d[ sab ]  for sab in  sab_for_lf])
            umls_labels = ontologyLF.OntologyLabelingFunction( text, token_flatten, spans, start_spans, terms, picos=None, expand_term=True, fuzzy_match=False )
            accumulated_lf.append( umls_labels )
            number_lfs.append( number )

        accumulated_labels.append( accumulated_lf )
        write2File(accumulated_lf, token_flatten, number_lfs, picos)

        #if i == 1:
        #    break

def label_umls_and_write(indir, umls_d, picos, text, token_flatten, spans, start_spans):

    if str(indir).split('/')[-1] == 'fuzzy':
        fuzzy_match = True
    else:
        fuzzy_match = False

    for k, v in umls_d.items():
        print( 'Fetching the labels for ', str(k) )
        umls_labels = ontologyLF.OntologyLabelingFunction( text, token_flatten, spans, start_spans, v, picos=None, expand_term=True, fuzzy_match=fuzzy_match )

        df = pd.DataFrame( {'tokens': token_flatten, str(k): umls_labels, })
        filename = 'lf_' + str(k) + '.tsv'
        df.to_csv(f'{indir}/{picos}/{filename}', sep='\t')


def label_ont_and_write(indir, terms, picos, text, token_flatten, spans, start_spans, ontology_name:str):

    if str(indir).split('/')[-1] == 'fuzzy':
        fuzzy_match = True
    else:
        fuzzy_match = False

    print( 'Fetching the labels for ', str(ontology_name) )
    nonumls_labels = ontologyLF.OntologyLabelingFunction( text, token_flatten, spans, start_spans, terms, picos=picos, expand_term=False, fuzzy_match=fuzzy_match )

    df = pd.DataFrame( {'tokens': token_flatten, str(ontology_name): nonumls_labels })
    filename = 'lf_' + str(ontology_name) + '.tsv'
    df.to_csv(f'{indir}/{picos}/{filename}', sep='\t')

