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
def extractAnnotation(source, target, match):
    
    token = list()
    annot = list()
    
    span_generator = WhitespaceTokenizer().span_tokenize(target)

    annotation_start_position = match[1][0]
    annotation_stop_position = match[1][0] + match[1][2] # start + stop position

    # print( source, ' : ' , annotation_start_position, ' : ', annotation_stop_position )
    # print( target )

    annotation = [0] * len(target)
    for n, i in enumerate(annotation):
        if n >= annotation_start_position and n <= annotation_stop_position: # if its anything between the start and the stop position of annotation
            annotation[n] = 1

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
# Function to align source intervention terms with high confidence short targets
###############################################################################################
def align_highconf_shorttarget(target, source):
    annot = list() # Get's updated for each Intervention name identified
    token = list()

    if target is not None:
        # Match the source to the target
        s = difflib.SequenceMatcher(None, target, source, autojunk=True)
        matches = fullMatchScore(s, source, target)
        for match in matches:
            if match[0] == 1.0:                    
                token, annot = extractAnnotation(source, target, match)

    assert len(token) == len(annot)
    return token, annot

###############################################################################################
# Function to align source intervention terms with high confidence long targets
###############################################################################################
def align_highconf_longtarget(target, source):

    target_sentences = list()
   
    if target is not None :
        # Sentence tokenization
        target_sentences = sent_tokenize(target)
        collect_annotations = dict()
        
        # Iterate each sentence
        for i, eachSentence in enumerate(target_sentences):

            annot = list() # Get's updated for each Intervention name identified and for each sentence
            token = list()

            s = difflib.SequenceMatcher(None, eachSentence, source, autojunk=True)
            matches = fullMatchScore(s, source, target)
            match_scores = [item[0] for item in matches ]

            if 1.0 in match_scores:
                for match in matches:
                    if match[0] == 1.0:  
                        token_i, annot_i = extractAnnotation(source, eachSentence, match)
                        annot.extend( annot_i )
                        token.extend( token_i )
                
            if annot:
                # print( annot )
                token_annot = [ token, annot ]
                collect_annotations['sentence' + str(i)] = token_annot

    assert len(token) == len(annot)

    if collect_annotations:
        print( collect_annotations )

    return collect_annotations