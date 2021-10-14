#!/usr/bin/env python
'''
This module contains functions to obtain block-wise match score and (summed) block-wise for the matched Source and Target texts.
'''

import sys, json, os
import datetime as dt
import time


def partMatchScore(s, source_term):
    match_score = sum( n for i,j,n in s.get_matching_blocks() ) / float(len(source_term)) # if this = 1.0, then entire Source text(intervention name) matched with the Target text
    return match_score

'''
Description:
    The funtion scores matches identified by the difflib.SequenceMatcher function. The scoring (match_score) is performed in a way that only when complete source term
    aligns with the target does the "match" receive a higher score.

Args:
    s (SequenceMatcher): dictionary storing "intervention" and "arms groups" terms
        source_term (string): source term string for scoring 
        target (string): target string for scoring 

Returns:
    list: returns a list of tuples containing [match score, matching block]
'''
def fullMatchScore(s, source_term, target):
    # match_score = sum( n for i,j,n in s.get_matching_blocks() ) / float(len(source_term))
    all_match_scores = []
    # Return the most confident block (most confident block is block / len(source_term) == 1.0)
    for eachMatchingBlock in s.get_matching_blocks()[:-1]: # Last block is a dummy block
        match_score = eachMatchingBlock[2] / float(len(source_term))
        score_block = [match_score, eachMatchingBlock]
        all_match_scores.append( score_block )

    return all_match_scores