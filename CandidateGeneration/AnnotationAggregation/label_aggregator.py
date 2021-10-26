
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
def intra_aggregate_labels(to_aggregate, aggregation_collector):

    # print( to_aggregate )

    intra_aggregator = dict()

    # print( 'before aggregation: ',  to_aggregate )

    for key, value in to_aggregate.items():
        for a_key, a_value in value.items():
            if a_key != 'source':
                if a_key not in intra_aggregator:
                    intra_aggregator[a_key] = a_value
                else:
                    # Get the key-value and aggregate at the level of value
                    for b_key, b_value in a_value.items():
                        if b_key in intra_aggregator[a_key]: # if the sentence IDs are identical aggregate at the sentence level

                            aggregate_this = b_value[1]
                            to_this = intra_aggregator[a_key][b_key][1]

                            for count, (i, j)  in enumerate(zip(aggregate_this, to_this)):
                                if i > j:
                                    intra_aggregator[a_key][b_key][1][count] = i
                        else: # if the sentence IDs are different and not yet in the intra-aggregation dictionary
                            intra_aggregator[a_key][b_key] = b_value


    # print( 'Intra aggregator: ', intra_aggregator )

    # Add the intra-aggregator content into inter-aggragator (aggregation_collector)
    isEmpty = bool(aggregation_collector)
    isIntra_aggregatorEmpty = bool(intra_aggregator)
    if isEmpty == False: # If aggregation_collector is empty then put everything from the intra_aggregator to it....
        aggregation_collector = intra_aggregator
    else:
        # if intra_aggregator is not empty, then aggregate intra_aggregator to the aggregation_collector....
        if isIntra_aggregatorEmpty == True:          

            for key, value in intra_aggregator.items():
                if key not in aggregation_collector:
                    aggregation_collector[key] = value
                else:
                    for a_key, a_value in value.items():
                        
                        if a_key in aggregation_collector[key]:
                            aggregate_this = a_value[1]
                            to_this = aggregation_collector[key][a_key][1]

                            for count, (i, j)  in enumerate(zip(aggregate_this, to_this)):
                                if i > j:
                                    aggregation_collector[key][a_key][1][count] = i
                        else:
                            aggregation_collector[key][a_key] = a_value

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

                    else:

                        print( source_key )

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
    label_resolver = load('/home/anjani/distant-PICO/CandidateGeneration/AnnotationAggregation/resolver_models/resolver.joblib') 

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
                        for eachHit in defects:
                            print( eachHit )

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