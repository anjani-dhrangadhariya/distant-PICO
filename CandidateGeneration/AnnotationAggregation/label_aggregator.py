
import collections
import enum
import re
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from joblib import dump, load
from sanity_checks import *
from AnnotationAggregation.label_resolver import *
from itertools import groupby

def global_annot_aggregator(to_aggregate):

    # print( to_aggregate )

    aggregated = dict()

    for operator in to_aggregate:
        
        for source_key, source in operator.items():
            
            for target_key, target in source.items():
                if 'source' not in target_key:
                    if target_key not in aggregated:

                        aggregated[target_key] = target

                    else:
                        # okay!! So if the target is already in the aggregated dictionary, could you check if each of the target sentence is there as well?

                        for sentence_key, sentence in target.items():

                            if sentence_key not in aggregated[target_key]:
                                
                                aggregated[target_key][sentence_key] = sentence
                            
                            elif sentence_key in aggregated[target_key]:

                                for annot_key, annot in sentence.items():

                                    if annot_key not in aggregated[target_key][sentence_key] and 'tokens' not in annot_key:

                                        aggregated[target_key][sentence_key][annot_key] = annot

                                    elif annot_key in aggregated[target_key][sentence_key] and 'tokens' not in annot_key: # if the annotations need to be merged....

                                        # make new annotations by merging
                                        merged_annot = [0] * len(annot)
                                        for counter, (o_a, n_a) in enumerate(zip( aggregated[target_key][sentence_key][annot_key], annot  )):
                                            chosen_annot = max( o_a, n_a )
                                            merged_annot[counter] = chosen_annot

                                        aggregated[target_key][sentence_key][annot_key] = merged_annot

    return aggregated


def intra_source_aggregator(to_aggregate):

    aggregated = dict()

    for key, sourceNumber in to_aggregate.items():
        for target_key, target in sourceNumber.items():
            if 'source' not in target_key and target_key not in aggregated: # The target is not present in the aggregated dictionary
                aggregated[target_key] = target
            elif 'source' not in target_key and target_key in aggregated: # The target is present in the aggregated dictionary

                for sentence_key, sentence in target.items():

                    if sentence_key in aggregated[target_key]: # if the sentence IDs are found in the aggregate dictionary

                        token_T, annot_T = aggregated[target_key][sentence_key]
                        token_S, annot_S = target[sentence_key]

                        assert len( token_S ) == len( token_T ) == len( annot_T ) == len( annot_S ) # Annotations being merged are identical lengths

                        annot_i = [0] * len( token_S )
                        for counter, (s,t) in enumerate(zip(annot_S, annot_T)):
                            chosen_value = max(s,t)
                            annot_i[counter] = chosen_value

                        aggregated[target_key][sentence_key] = [token_T, annot_i]

                    elif sentence_key not in aggregated[target_key]:  # if the sentence IDs are missing from the aggregate dictionary
                        aggregated[target_key][sentence_key] = sentence

    return aggregated

def intra_operator_aggregator(to_aggregate, aggregation_collector):

    for key, target in to_aggregate.items():
        if key not in aggregation_collector:
            aggregation_collector[key] = target
        else:
            print( 'target nathi: ',  target )

    return aggregation_collector


def restructureAnnot(annot, annot_type):

    temp_annot = dict()

    temp_annot = annot
    for key, target in annot.items():
        for a_key, sentence in target.items():
            if a_key not in temp_annot:
                temp_annot[key][a_key] = { 'tokens': sentence[0] }
                temp_annot[key][a_key][annot_type] = sentence[1]
                assert len( sentence[0] ) == len( sentence[1] )
            else:
                temp_annot[key][a_key][annot_type] = sentence[1]
                assert len( temp_annot[key][a_key][annot_type]['tokens'] ) == len( sentence[1] ) 

    return temp_annot

''' TODO
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
def inter_aggregate_labels(p, ic, o, s, inter_aggregator):

    temp_p = dict()
    temp_ic = dict()
    temp_o = dict
    temp_s = dict()

    # To restructure the individual dictionaries
    temp_p = restructureAnnot(p, 'p')
    temp_ic = restructureAnnot(ic, 'ic')
    temp_o = restructureAnnot(o, 'o')
    temp_s = restructureAnnot(s, 's')

    temp_intermediate = [temp_p, temp_ic, temp_o, temp_s]

    for d in temp_intermediate: # Iterate each dictionary in the list of dictionaries

        for k, target in d.items():

            if k not in inter_aggregator:
                inter_aggregator[k] = dict()

            for k_i, sentence in target.items():

                if k_i not in inter_aggregator[k]:
                    inter_aggregator[k][k_i] = sentence

                else:
                    resultset_source = [ [key, value] for key, value in sentence.items() if key not in ['tokens']]
                    resultset_target = [ [key, value] for key, value in inter_aggregator[k][k_i].items() if key not in ['tokens']]

                    source_key = resultset_source[0][0]
                    if source_key not in inter_aggregator[k][k_i]:

                        inter_aggregator[k][k_i][source_key] = resultset_source[0][1]
                        # if len( resultset_target[0][1] ) != len( resultset_source[0][1] ):
                        #     print( inter_aggregator[k][k_i]['tokens'] )
                        #     print( resultset_target[0][1] )
                        #     print( sentence['tokens'] )
                        #     print( resultset_source[0][1] )
                    # else:
                    #     print( source_key )

    # Perform sanity check before returning the aggregated dictionary 
    sanity_check_globalagg(temp_p, temp_ic, temp_o, temp_s, inter_aggregator)
    
    return inter_aggregator


''' TODO
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
def merge_labels(globally_aggregated):

    globally_merged = dict()

    # Load the resolver model
    label_resolver = load('/home/anjani/distant-PICO/CandidateGeneration/AnnotationAggregation/resolver_models/resolver_large.joblib') 

    # Iterate through the labels here
    for key, target in globally_aggregated.items():

        if key not in globally_merged:
            globally_merged[key] = {}

        if 'id' not in key: # do not iterate the NCT_ID
            for a_key, sentence in target.items():

                # Merge and collect the annotations here
                annotations = [0] * len(sentence['tokens'])

                if a_key not in globally_merged[key]:
                    globally_merged[key][a_key] = {}
                    globally_merged[key][a_key]['tokens'] = sentence['tokens']

                if len(sentence) == 2: # No resolution needed for this case
                    annot_index = list(sentence.keys())[-1]
                    annotations = sentence[annot_index]
                    globally_merged[key][a_key]['annotation'] = annotations

                
                elif len(sentence) >= 3: # resolution needed for this case as there could be some overlap

                    resultset = [ value for key, value in sentence.items() if key not in ['tokens']]

                    len_first = len(resultset[0]) if resultset else None
                    if all(len(i) == len_first for i in resultset) == False:
                        defects = groupby(sorted(resultset, key=len), key=len)
                        # for eachHit in defects:
                        #     print( eachHit )

                    assert all(len(i) == len_first for i in resultset) == True # Check if all the annotation lenths are identical

                    phrase = []
                    counter_values = []
                    overlap = []

                    for counter, i in enumerate(zip(*resultset))  :
                        if sum([1 if n else 0 for n in i])  > 1:
                            phrase.append( globally_merged[key][a_key]['tokens'][counter] )
                            counter_values.append( counter )
                            if 0 in i:
                                temp_i = list(i)
                                temp_i.sort()
                                temp_i =  temp_i[1:]
                                overlap.append( tuple( temp_i ) )
                            else:
                                overlap.append( i )
                        elif sum([1 if n else 0 for n in i])  == 1:
                            temp_i = list(i)
                            temp_i.sort()
                            annotations[counter] = temp_i[-1]

                    if phrase:
                        label_resolved = label_resolver.predict( phrase )
                        label_resolved_proba = label_resolver.predict_proba( phrase )

                        for word, over, counter, l, l_p in zip(phrase, overlap, counter_values, label_resolved, label_resolved_proba):
                            if l in over:
                                annotations[counter] = l
                            else:
                                
                                temp_i = list(over)
                                chosen_index = [ l_p[ n-1 ] for n in temp_i]
                                chosen_annot = max( chosen_index )
                                annotations[counter] = list(l_p).index(chosen_annot) + 1
                                # print( word, over, counter, l, l_p, chosen_annot )

    return globally_merged