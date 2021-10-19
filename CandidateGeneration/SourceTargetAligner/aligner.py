#!/usr/bin/env python
'''
This module contains functions to align source intervention terms to the target sentences (short targets, long targets - with and without negative candidates). It also has function to extract the annotations once source terms are aligned to the target.
'''

import sys, json, os
import datetime as dt
import time
import random

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search,  Q

import difflib, re
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

####################################################################
# Function to extract the aligned candidate annotations
####################################################################
def extractReGeXannotation(source, target, match, PICOS):

    token = list()
    annot = list()
    
    span_generator = WhitespaceTokenizer().span_tokenize(target)

    annotation_start_position = match.start()
    annotation_stop_position = match.end()

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

    # Check if the number of annotations match number of tokens present in the sentence
    assert len(token) == len(annot)
       
    return token, annot


def extractAnnotation(source, target, match, PICOS):
    
    token = list()
    annot = list()
    
    span_generator = WhitespaceTokenizer().span_tokenize(target)

    annotation_start_position = match[1][0]
    annotation_stop_position = match[1][0] + match[1][2] # start + stop position

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

    # Check if the number of annotations match number of tokens present in the sentence
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

            # print('------------------', key )

            annot = list()
            token = list()
            pos = list()
            pos_fine = list()

            eachSentence = value['text'].lower()
            eachSentence_tokens = value['tokens']
            eachSentence_pos = value['pos']
            eachSentence_posfine = value['pos_fine']

            s = difflib.SequenceMatcher(None, eachSentence, source, autojunk=True)
            matches = fullMatchScore(s, source, target)
            match_scores = [item[0] for item in matches ]

            if 1.0 in match_scores:
                for match in matches:
                    if match[0] == 1.0:  
                        token_i, annot_i = extractAnnotation(source, eachSentence, match, PICOS)
                        annot.extend( annot_i )
                        token.extend( token_i )
                        pos.extend( eachSentence_pos )
                        pos_fine.extend( eachSentence_posfine )
                        # print( len(token), len(annot), len(eachSentence_pos), len(eachSentence_posfine) ) # TODO: We are not extending them because the lengths of tokens and annot do not correspond to the length of pos and pos_fine

            if annot:
                token_annot = [ token, annot ]
                # token_annot = [ token, annot, eachSentence_pos, eachSentence_posfine ]
                collect_annotations[key] = token_annot

            assert len(token) == len(annot)

    return collect_annotations

###############################################################################################
# Function to align source ReGeX to long targets
###############################################################################################
def align_regex_longtarget(target, source, PICOS):
 
    target_sentences = list()

    if target is not None :
        # Sentence tokenization
        # target_sentences = sent_tokenize(target)
        collect_annotations = dict()
       
        # Iterate each sentence
        # for i, eachSentence in enumerate(target_sentences):
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
                
                token_i, annot_i = extractReGeXannotation(source, eachSentence, match, PICOS)
                annot.extend( annot_i )
                token.extend( token_i )
                pos.extend( eachSentence_pos )
                pos_fine.extend( eachSentence_posfine )
                # print( len(token), len(annot), len(eachSentence_pos), len(eachSentence_posfine) ) # TODO: We are not extending them because the lengths of tokens and annot do not correspond to the length of pos and pos_fine

            if annot:
                temp = []
                for t, a, in zip(token, annot):
                    if a != 0:
                        temp.append( t )
                temp = ' '.join(temp)

                token_annot = [ token, annot ]
                # token_annot = [ token, annot, eachSentence_pos, eachSentence_posfine ]
                collect_annotations[key] = [temp, token_annot]

            assert len(token) == len(annot)

    return collect_annotations