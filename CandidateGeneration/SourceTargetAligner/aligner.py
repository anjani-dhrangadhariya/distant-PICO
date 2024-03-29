#!/usr/bin/env python
'''
This module contains functions to align source intervention terms to the target sentences (short targets, long targets - with and without negative candidates). It also has function to extract the annotations once source terms are aligned to the target.
'''

from itertools import count
import sys, json, os
import datetime as dt
import time
import random

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search,  Q

import difflib, re
import spacy
import nltk
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize, word_tokenize

from collections import Counter
from collections import defaultdict
import collections
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pylab import *

from SourceTargetAligner.scoring import *
from CandGenUtilities.labeler_utilities import *

nlp = spacy.load("en_core_sci_sm")

####################################################################
# Function to extract the aligned candidate annotations
####################################################################
def extractAnnotation(source, target, match, PICOS, isReGeX):
    
    token = list()
    annot = list()
    
    span_generator = WhitespaceTokenizer().span_tokenize(target)

    if isReGeX == True:
        annotation_start_position = match.start()
        annotation_stop_position = match.end()
    else:
        annotation_start_position = match[0]
        annotation_stop_position = match[1]

    annotation = [0] * len(target)
    for n, i in enumerate(annotation):
        if n >= annotation_start_position and n <= annotation_stop_position: # if its anything between the start and the stop position of annotation
            annotation[n] = PICOS

    for span in span_generator:
        # span for each token is generated here
        token_ = target[span[0]:span[1]]
        
        annot_ = annotation[span[0]:span[1]]
        
        max_element_i = Counter(annot_)
        max_element = max_element_i.most_common(1)

        token.append(token_)
        annot.append(max_element[0][0])

    assert len(token) == len(annot)
       
    return token, annot

###############################################################################################
# Label function for direct alignment
###############################################################################################
def direct_abstain(target_sentence, source, pos, PICOS):

    temp_annot = []

    source = source.split(' ')
    for s_i, p_i in zip(source, pos):

        if re.search( s_i, target_sentence ) and p_i in ['NOUN', 'PROPN', 'ADJ'] and s_i not in punct():

            matches = re.finditer(s_i, target_sentence)

            for m in matches:
                token_i, annot_i = extractAnnotation(s_i, target_sentence, m, PICOS, isReGeX=True)
                if len(temp_annot) == 0:
                    temp_annot = annot_i
                else:
                    for counter, (o_a, n_a) in enumerate(zip( temp_annot, annot_i )):
                        selected_annot = min( int(o_a), int(n_a) )
                        temp_annot[counter] = selected_annot

    return temp_annot

def direct_alignment(target, source, PICOS, to_abstain):

    if target is not None :
     
        # Initialize annotations
        token = target['tokens']
        annot = [0] * len( target['tokens'] )

        source_text = source['text'].replace('(', '\(').replace(')', '\)').lower()
        source_pos = source['pos']

        target_sentence = target['text'].lower()
        target_sentence_tokens = target['tokens']
        target_sentence_pos = target['pos']
        target_sentence_posfine = target['pos_fine']

        if re.search( source_text, target_sentence ): # The source is in the target

            matches = re.finditer(source_text, target_sentence)

            for m in matches:
                token_i, annot_i = extractAnnotation(source_text, target_sentence, m, PICOS, isReGeX=True)

                for counter, (o_a, n_a) in enumerate(zip( annot, annot_i )):
                    selected_annot = max( int(o_a), int(n_a) )
                    annot[counter] = selected_annot

            if to_abstain == True and len(source_text.split(' ')) > 1:
                annot_j = direct_abstain(target_sentence, source_text, source_pos, PICOS = -1)
                
                for counter, (o_a, n_a) in enumerate(zip( annot, annot_j )):
                    if n_a == -1 and o_a == 0:
                        annot[counter] = n_a

        else: # The source is not in the target
            
            if to_abstain == True and len(source_text.split(' ')) > 1:
                annot_j = direct_abstain(target_sentence, source_text, source_pos, PICOS = -1)

                for counter, (o_a, n_a) in enumerate(zip( annot, annot_j )):
                    if n_a == -1 and o_a == 0:
                        annot[counter] = n_a


    assert len(token) == len(annot)
    return token, annot


###############################################################################################
# Function to align source terms with high confidence long targets
###############################################################################################
def align_highconf_longtarget(target, source, PICOS):

    target_sentences = list()
   
    if target is not None :
        # Sentence tokenization
        collect_annotations = dict()
     
        # Iterate each sentence
        for key, value in target.items():

            annot = list()
            token = list()
            pos = list()
            pos_fine = list()

            eachSentence = value['text'].lower()
            eachSentence_tokens = value['tokens']
            eachSentence_pos = value['pos']
            eachSentence_posfine = value['pos_fine']

            # s = difflib.SequenceMatcher(None, eachSentence, source, autojunk=True)
            # matches = fullMatchScore(s, source, target)
            # match_scores = [item[0] for item in matches ]

            # if 1.0 in match_scores:
            #     for match in matches: # If one sentence has multiple matches....
            #         if match[0] == 1.0:
            #             token_i, annot_i = extractAnnotation(source, eachSentence, match, PICOS, isReGeX=False)
            #             if not annot and not token: # extend if the lists are empty
            #                 annot.extend( annot_i )
            #                 token.extend( token_i )
            #                 pos.extend( eachSentence_pos )
            #                 pos_fine.extend( eachSentence_posfine )
            #             else: # else if the lists are not empty, merge the annotations
            #                 for counter, (o_a, n_a) in enumerate(zip( annot, annot_i )):
            #                     selected_annot = max( o_a, n_a )
            #                     annot[counter] = selected_annot

            r1 = re.finditer(source, eachSentence)
            for match in r1:
                
                token_i, annot_i = extractAnnotation(source, eachSentence, match, PICOS, isReGeX=True)
                if not annot and not token: # if the lists are empty
                    annot.extend( annot_i )
                    token.extend( token_i )
                    pos.extend( eachSentence_pos )
                    pos_fine.extend( eachSentence_posfine )
                else:
                    for counter, (o_a, n_a) in enumerate(zip( annot, annot_i )):
                        selected_annot = max( o_a, n_a )
                        annot[counter] = selected_annot

            if annot:
                token_annot = {'tokens':token, str(PICOS): annot }
                collect_annotations[key] = token_annot
            else:
                abstain_annot = [-1] * len(value['tokens']) # Create "ABSTAIN" labels for these sentences
                token_annot = {'tokens': value['tokens'], str(PICOS): abstain_annot }
                collect_annotations[key] = token_annot

            assert len(token) == len(annot)

    return collect_annotations

###############################################################################################
# Function to align source ReGeX to long targets
###############################################################################################
def align_regex_longtarget(target, source, PICOS):
 
    target_sentences = list()

    if target is not None :
        collect_annotations = dict()
       
        # Iterate each sentence
        for key, value in target.items():

            eachSentence = value['text'].lower()
            eachSentence_tokens = value['tokens']
            eachSentence_pos = value['pos']
            eachSentence_posfine = value['pos_fine']

            annot = list()
            token = list()
            pos = list()
            pos_fine = list()

            source = re.compile(source)
            r1 = source.finditer(eachSentence)

            for match in r1:
                
                token_i, annot_i = extractAnnotation(source, eachSentence, match, PICOS, isReGeX=True)
                if not annot and not token: # if the lists are empty
                    annot.extend( annot_i )
                    token.extend( token_i )
                    pos.extend( eachSentence_pos )
                    pos_fine.extend( eachSentence_posfine )
                else:
                    for counter, (o_a, n_a) in enumerate(zip( annot, annot_i )):
                        selected_annot = max( o_a, n_a )
                        annot[counter] = selected_annot

            if annot:
                temp = []
                for t, a, in zip(token, annot):
                    if a != 0:
                        temp.append( t )
                temp = ' '.join(temp)

                token_annot = {'tokens':token, str(PICOS): annot }
                collect_annotations[key] = [temp, token_annot]

    return collect_annotations