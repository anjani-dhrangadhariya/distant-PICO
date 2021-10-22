
import collections
from sanity_checks import *
from AnnotationAggregation.label_resolver import *

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

    temp_p = p
    for key, value in temp_p.items():
        if key not in inter_aggregator:
            inter_aggregator[key] = dict()
        for a_key, a_value in value.items():
            temp_p[key][a_key] = { 'tokens': a_value[0] }
            temp_p[key][a_key]['p'] = a_value[1]

    temp_ic = ic
    for key, value in temp_ic.items():
        if key not in inter_aggregator:
            inter_aggregator[key] = dict()
        for a_key, a_value in value.items():
            temp_ic[key][a_key] = { 'tokens': a_value[0] }
            temp_ic[key][a_key]['ic'] = a_value[1]

    temp_o = o
    for key, value in temp_o.items():
        if key not in inter_aggregator:
            inter_aggregator[key] = dict()
        for a_key, a_value in value.items():
            temp_o[key][a_key] = { 'tokens': a_value[0] }
            temp_o[key][a_key]['o'] = a_value[1]

    temp_s = s
    for key, value in temp_s.items():
        if key not in inter_aggregator:
            inter_aggregator[key] = dict()
        for a_key, a_value in value.items():
            temp_s[key][a_key] = { 'tokens': a_value[0] }
            temp_s[key][a_key]['s'] = a_value[1]

    
    temp_intermediate = [temp_p, temp_ic, temp_o, temp_s]

    for d in temp_intermediate: # Iterate each dictionary in the list of dictionaries

        for k, v in d.items():
            for k_i, v_i in v.items():

                if k_i not in inter_aggregator[k]:
                    inter_aggregator[k][k_i] = v_i
                else:
                    inter_aggregator[k][k_i].update(v_i)

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
def merge_labels(p, ic, o, s, inter_aggregator):

    # Iterate through the labels here and whenever there is an overlap, resolve using the label resolver.
    


    return None