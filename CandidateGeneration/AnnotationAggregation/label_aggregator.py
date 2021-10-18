
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

def inter_aggregate_util(intra_i, inter, entity):

    for key, value in intra_i.items():
        if key not in inter:
            inter[key] = {}

    for key, value in intra_i.items():
        values = inter[key]
        for a_key, a_value in value.items():
            if a_key not in values:
                inter[key][a_key] = { 'tokens': a_value[0] }
                inter[key][a_key][entity] = a_value[1]
            else:
                inter[key][a_key][entity] = a_value[1]

    
    return inter


def inter_aggregate_labels(p, ic, o, s, inter_aggregator):

    # inter_aggregator = p
    # for key, value in inter_aggregator.items():
    #     for a_key, a_value in value.items():
    #         inter_aggregator[key][a_key] = { 'tokens': a_value[0] }
    #         inter_aggregator[key][a_key]['p'] = a_value[1]

    inter_aggregator = inter_aggregate_util(p, inter_aggregator, 'p')
    inter_aggregator = inter_aggregate_util(ic, inter_aggregator, 'ic')
    inter_aggregator = inter_aggregate_util(o, inter_aggregator, 'o')
    inter_aggregator = inter_aggregate_util(s, inter_aggregator, 's')

    return inter_aggregator