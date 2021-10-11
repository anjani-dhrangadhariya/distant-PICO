#!/usr/bin/env python

def LabelingOperatorDoc(a):
    '''The module includes labeling operators for different sources.'''
    return a**a

print( LabelingOperatorDoc.__doc__ )

from SourceTargetAligner.aligner import *

'''
Direct Aligner
'''
def directAligner(source, targets, candidateTargets):

    combined_annotations = dict()

    for eachSource in source: # each source_i

        for key, value in targets.items(): # each target_i

            res = key.startswith(tuple(candidateTargets))
            if res == True:
                target_i = targets[key]['text']
                annotations = align_highconf_longtarget( target_i.lower() , eachSource.lower() )

                if annotations:
                    if key not in combined_annotations:
                        combined_annotations[key] = [annotations]
                    else:
                        combined_annotations[key].append( annotations )

    return combined_annotations

'''
ReGeX Aligner
'''
def regexAligner(source, targets, candidateTargets):

    combined_annotations = dict()

    for eachReGeX in source: # each source_i

        # for eachCandidate in candidateTargets: # each target_i
        for key, value in targets.items(): # each target_i

            res = key.startswith(tuple(candidateTargets))
            if res == True:

                target_i = targets[key]['text']

                annotations = align_regex_longtarget( target_i.lower() , eachReGeX )

                if annotations:
                    if key not in combined_annotations:
                        combined_annotations[key] = [annotations]
                    else:
                        combined_annotations[key].append( annotations )
    
    return combined_annotations

'''
LongTail Intervention Aligner
'''
def longTailInterventionAligner(source, targets, candidateTargets):

    intervention_annotations = dict()

    for eachKey, eachValue in source.items(): # each source_i

        intervention_term = list(eachValue)[0]['text']

        for key, value in targets.items(): # each target_i

            res = key.startswith(tuple(candidateTargets))
            if res == True:
                target_i = targets[key]['text']

                annotations = align_highconf_longtarget( target_i.lower() , intervention_term.lower() )

                if annotations:
                    if key not in intervention_annotations:
                        intervention_annotations[key] = [annotations]
                    else:
                        intervention_annotations[key].append( annotations )

    return intervention_annotations

'''
LongTail Condition Aligner
'''
def longTailConditionAligner(source, targets, candidateTargets):

    condition_annotations = dict()

    for eachCondition in source: # each source_i

        for key, value in targets.items(): # each target_i

            res = key.startswith(tuple(candidateTargets))
            if res == True:        

                target_i = targets[key]['text']
                annotations = align_highconf_longtarget( target_i.lower() , eachCondition.lower() )

                if annotations:
                    if key not in condition_annotations:
                        condition_annotations[key] = [annotations]
                    else:
                        condition_annotations[key].append( annotations )

    return condition_annotations