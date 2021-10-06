#!/usr/bin/env python

def ParticipantAlignerDoc(a):
    '''The module generates scored alignment between Participant sources and relevant targets.'''
    return a**a

print( ParticipantAlignerDoc.__doc__ )

from SourceTargetAligner.aligner import *

def directAligner(source, targets, candidateTargets):

    combined_annotations = dict()

    for eachSource in source: # each source_i

        for eachCandidate in candidateTargets: # each target_i

            target_i = targets[eachCandidate]['text']
            annotations = align_highconf_longtarget( target_i.lower() , eachSource.lower() )

            if annotations:
                if eachCandidate not in combined_annotations:
                    combined_annotations[eachCandidate] = [annotations]
                else:
                    combined_annotations[eachCandidate].append( annotations )

    return combined_annotations

def longTailAligner(source, targets, candidateTargets):

    condition_annotations = dict()

    for eachCondition in source: # each source_i

        for eachCandidate in candidateTargets: # each target_i

            target_i = targets[eachCandidate]['text']
            annotations = align_highconf_longtarget( target_i.lower() , eachCondition.lower() )

            if annotations:
                if eachCandidate not in condition_annotations:
                    condition_annotations[eachCandidate] = [annotations]
                else:
                    condition_annotations[eachCandidate].append( annotations )

    return condition_annotations