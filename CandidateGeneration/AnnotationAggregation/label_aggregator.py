
def aggregate_labels(to_aggregate, aggregation_collector):

    intra_aggregator = dict()

    # print( 'before aggregation: ',  to_aggregate )

    # print( aggregation_collector )

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


    print( 'Intra aggregator: ', intra_aggregator )

    print('############################## END of NCT ID ######################################')
    # Intra aggregation