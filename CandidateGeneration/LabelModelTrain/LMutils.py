import pandas as pd
from sklearn.model_selection import train_test_split
import json

def list2Nested(l, nested_length):
    return [l[i:i+nested_length] for i in range(0, len(l), nested_length)]

def partitionRankedSAB(umls_d):

    keys = list(umls_d.keys())

    partitioned_lfs = [ ]

    for i in range( 0, len(keys) ):

        if i == 0 or i == len(keys):
            if i == 0:
                partitioned_lfs.append( [keys] )
            if i ==len(keys):
                temp3 = list2Nested(keys, 1)
                partitioned_lfs.append( temp3 )
        else:
            temp1, temp2 = keys[:i] , keys[i:]
            temp3 = list2Nested( keys[:i], 1)
            temp3.append( keys[i:] )
            partitioned_lfs.append( temp3 )

    return partitioned_lfs


def rankSAB(umls_d, picos):

    ranked_umls = []
    ranked_dict = dict()

    if picos == 'p':
        ranks_p = open('/home/anjani/distant-PICO/CandidateGeneration/Ontologies/umls_p_rank.txt','r').read()
        ranked_umls = eval(ranks_p)
    if picos == 'i':
        ranks_i = open('/home/anjani/distant-PICO/CandidateGeneration/Ontologies/umls_i_rank.txt','r').read()
        ranked_umls = eval(ranks_i)
    if picos == 'o':
        ranks_o = open('/home/anjani/distant-PICO/CandidateGeneration/Ontologies/umls_o_rank.txt','r').read()
        ranked_umls = eval(ranks_o)

    for i, l in enumerate(ranked_umls):
        if l[0] in umls_d:
            ranked_dict[ l[0] ] = umls_d[l[0]]

    partitioned_umls = partitionRankedSAB(ranked_dict)

    return ranked_dict, partitioned_umls