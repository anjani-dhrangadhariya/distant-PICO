#!/usr/bin/env python

def IntSourceFetcherDoc(a):
    '''The module retrieves sources for each of the I - Intervention and C - Comparator entities'''
    return a**a

print( IntSourceFetcherDoc.__doc__ )


def fetchIntcompSources(json_document):

    combined_sources = dict()

    p_condition = fetchParticipantCondition(json_document)

    return combined_sources