#!/usr/bin/env python
'''
This module contains functions to obtain block-wise match score and (summed) block-wise for the matched Source and Target texts.
'''

import sys, json, os
import datetime as dt
import time


def partMatchScore(s, interventionName):
    match_score = sum( n for i,j,n in s.get_matching_blocks() ) / float(len(interventionName)) # if this = 1.0, then entire Source text(intervention name) matched with the Target text
    return match_score

def fullMatchScore(s, interventionName, briefTitleTarget):
    # match_score = sum( n for i,j,n in s.get_matching_blocks() ) / float(len(interventionName))
    all_match_scores = []
    # Return the most confident block (most confident block is block / len(interventionName) == 1.0)
    for eachMatchingBlock in s.get_matching_blocks()[:-1]: # Last block is a dummy block
        match_score = eachMatchingBlock[2] / float(len(interventionName))
        score_block = [match_score, eachMatchingBlock]
        all_match_scores.append( score_block )
        # yield score_block
    return all_match_scores