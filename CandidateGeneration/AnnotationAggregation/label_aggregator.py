
def aggregate_labels(to_aggregate, aggregation_collector):

    print( to_aggregate )

    for key, annotation_main in to_aggregate.items():
        
        for a_key, a_value in annotation_main.items():

            if 'source' not in a_key:
                if a_key in aggregation_collector:
                    print('########################################################')
                    aggregation_collector[a_key].append( a_value[0] )
                    print( aggregation_collector[a_key] )

        # print( aggregation_collector )