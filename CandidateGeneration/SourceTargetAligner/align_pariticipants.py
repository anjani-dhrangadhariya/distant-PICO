#!/usr/bin/env python

def ParticipantAlignerDoc(a):
    '''The module generates scored alignment between Participant sources and relevant targets.'''
    return a**a

print( ParticipantAlignerDoc.__doc__ )

from SourceTargetAligner.aligner import *

def alignParGender(source, targets, candidateTargets):

    gender_annotations = dict()

    for eachGender in source: # each source_i

        for eachCandidate in candidateTargets: # each target_i

            target_i = targets[eachCandidate]['text']
            annotations = align_highconf_longtarget( target_i.lower() , eachGender.lower() )

            if annotations:
                if eachCandidate not in gender_annotations:
                    gender_annotations[eachCandidate] = [annotations]
                else:
                    gender_annotations[eachCandidate].append( annotations )

    return gender_annotations

def alignParSampSize(source, targets, candidateTargets):

    sampsize_annotations = []

    for eachCandidate in candidateTargets: # each target_i

        target_i = targets[eachCandidate]['text']
        annotations = align_highconf_longtarget( target_i.lower() , source )

        if annotations:
            sampsize_annotations.append(annotations)

    return sampsize_annotations