#!/usr/bin/env python

def LabelingOperatorDoc(a):
    '''The module includes labeling operators for different sources.'''
    return a**a

print( LabelingOperatorDoc.__doc__ )

import enum
from SourceTargetAligner.aligner import *

'''
Description:
    The function directly aligns a list of concepts from the source dictionary to the appropriate targets

Args:
    source (list): list to candidate concepts to be aligned with the targets
        targets (dict): A dictionary with all the targets for distant-PICOS
        candidateTargets (dict): A dictionary to select appropriate targets for each of the PICOS sources
        PICOS (int): Label for the entity being weakly annotations

Returns:
    dictionary: weakly annotated sources with the givens sources and PICOS labeling scheme
'''
def directAligner(source, targets, candidateTargets, PICOS):

    counter = 0
    combined_annotations = dict()

    for i, eachSource in enumerate(source): # each source_i

        for key, value in targets.items(): # each target_i

            res = key.startswith(tuple(candidateTargets))
            if res == True:
                target_i = targets[key]
                annotations = align_highconf_longtarget( target_i , eachSource.lower() , PICOS)

                if annotations:
                    if eachSource not in combined_annotations:
                        combined_annotations[str(counter)] = {'source': eachSource}

                    if key not in combined_annotations[str(counter)]:
                        combined_annotations[str(counter)][key] = [annotations]
                    else:
                        combined_annotations[str(counter)][key].append( annotations )

                    counter = counter + 1

    return combined_annotations

'''
Description:
    The function directly aligns a list of regular expressions from the source list to the appropriate targets

Args:
    source (list): list to candidate regular expressions (ReGeX) to be aligned with the targets
        targets (dict): A dictionary with all the targets for distant-PICOS
        candidateTargets (dict): A dictionary to select appropriate targets for each of the PICOS sources
        PICOS (int): Label for the entity being weakly annotations

Returns:
    dictionary: weakly annotated sources with the givens sources and PICOS labeling scheme
'''
def regexAligner(source, targets, candidateTargets, PICOS):

    combined_annotations = dict()

    for i, eachReGeX in  enumerate(source): # each source_i

        for key, value in targets.items(): # each target_i

            res = key.startswith(tuple(candidateTargets))
            if res == True:

                target_i = targets[key]

                annotations = align_regex_longtarget( target_i , eachReGeX , PICOS)

                if annotations:
                    matching_source = list(annotations.values())[0][0]
                    if str(i) not in combined_annotations:
                        combined_annotations[str(i)] = {'source': matching_source}

                    new_annotations = dict() # Remove the sources from original annotations
                    for new_key, new_value in annotations.items():
                        new_annotations[new_key] = new_value[1:]

                    if key not in combined_annotations[str(i)]:
                        combined_annotations[str(i)][key] = [new_annotations]
                    else:
                        combined_annotations[str(i)][key].append( new_annotations )

    return combined_annotations

'''
Description:
    The function directly aligns a list of "intervention" terms from the source list to the appropriate targets

Args:
    source (dictionary): a dictionary of candidate intervention terms to be aligned with the targets
        targets (dict): A dictionary with all the targets for distant-PICOS
        candidateTargets (dict): A dictionary to select appropriate targets for each of the PICOS sources
        PICOS (int): Label for the entity being weakly annotations

Returns:
    dictionary: weakly annotated sources with the givens sources and PICOS labeling scheme
'''
def longTailInterventionAligner(source, targets, candidateTargets, PICOS):

    intervention_annotations = dict()

    for i, (eachKey, eachValue) in enumerate(source.items()): # each source_i

        for eachValue_i in eachValue:

            intervention_term = eachValue_i['text']

            if intervention_term not in intervention_annotations:
                intervention_annotations[str(i)] = {'source': intervention_term}

            for key, value in targets.items(): # each target_i

                res = key.startswith(tuple(candidateTargets))
                if res == True:
                    target_i = targets[key]

                    annotations = align_highconf_longtarget( target_i, intervention_term.lower() , PICOS)

                    if annotations:
                        if key not in intervention_annotations[str(i)]:
                            intervention_annotations[str(i)][key] = [annotations]
                        else:
                            intervention_annotations[str(i)][key].append( annotations )

    return intervention_annotations

'''
Description:
    The function directly aligns a list of "Participant: Condition" terms from the source list to the appropriate targets

Args:
    source (list): list to candidate participant condition terms to be aligned with the targets
        targets (dict): A dictionary with all the targets for distant-PICOS
        candidateTargets (dict): A dictionary to select appropriate targets for each of the PICOS sources
        PICOS (int): Label for the entity being weakly annotations

Returns:
    dictionary: weakly annotated sources with the givens sources and PICOS labeling scheme
'''
def longTailConditionAligner(source, targets, candidateTargets, PICOS):

    condition_annotations = dict()

    for i, eachCondition in enumerate(source): # each source_i

        if eachCondition not in condition_annotations:
            condition_annotations[str(i)] = {'source': eachCondition}

        for key, value in targets.items(): # each target_i

            res = key.startswith(tuple(candidateTargets))
            if res == True:        

                target_i = targets[key]
                annotations = align_highconf_longtarget( target_i , eachCondition.lower() , PICOS)

                if annotations:
                    if key not in condition_annotations[str(i)]:
                        condition_annotations[str(i)][key] = [annotations]
                    else:
                        condition_annotations[str(i)][key].append( annotations )
    
    return condition_annotations

'''
Description:
    The function directly aligns a list of "Outcome: (Primary and Secondary)" terms from the source list to the appropriate targets

Args:
    source (dict): dictionary of candidate outcome terms to be aligned with the targets
        targets (dict): A dictionary with all the targets for distant-PICOS
        candidateTargets (dict): A dictionary to select appropriate targets for each of the PICOS sources
        PICOS (int): Label for the entity being weakly annotations

Returns:
    dictionary: weakly annotated sources with the givens sources and PICOS labeling scheme
'''
def longTailOutcomeAligner(source, targets, candidateTargets, PICOS):

    outcome_annotations = dict()

    for i, (eachKey, eachValue) in enumerate(source.items()): # each source_i

        for eachValue_i in eachValue:

            outcome_term = eachValue_i['text']

            if outcome_term not in outcome_annotations:
                outcome_annotations[str(i)] = {'source': outcome_term}

            for key, value in targets.items(): # each target_i

                res = key.startswith(tuple(candidateTargets))
                if res == True:
                    target_i = targets[key]

                    annotations = align_highconf_longtarget( target_i, outcome_term.lower() , PICOS)

                    if annotations:
                        if key not in outcome_annotations[str(i)]:
                            outcome_annotations[str(i)][key] = [annotations]
                        else:
                            outcome_annotations[str(i)][key].append( annotations )

    return outcome_annotations