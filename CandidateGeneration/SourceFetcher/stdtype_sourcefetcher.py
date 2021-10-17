#!/usr/bin/env python

def StdtypeSourceFetcherDoc(a):
    '''The module retrieves sources of the S - Study type entity'''
    return a**a

print( StdtypeSourceFetcherDoc.__doc__ )


'''
Description:
    Extracts the string of study type information of the clinical trial from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    string: a string containing study type information in the clinical trials
'''
def fetchStudyType(json_document):

    studyType = 'N/A'

    if 'DesignModule' in json_document:
            if 'DesignInfo' in json_document['DesignModule']:
                if 'DesignAllocation' in json_document['DesignModule']['DesignInfo']:
                    studyType = json_document['DesignModule']['DesignInfo']['DesignAllocation']

    return studyType 


def fetchStdTypeSources(json_document):

    combined_sources=dict()

    studyType = fetchStudyType(json_document)
    combined_sources['s_type'] = studyType

    return combined_sources