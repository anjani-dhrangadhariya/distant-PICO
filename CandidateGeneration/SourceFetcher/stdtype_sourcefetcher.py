#!/usr/bin/env python

def StdtypeSourceFetcherDoc(a):
    '''The module retrieves sources of the S - Study type entity'''
    return a**a

print( StdtypeSourceFetcherDoc.__doc__ )

from SourceTargetExpander.SourceExpander.expansion_utils import *

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

    studyType = dict()

    if 'DesignModule' in json_document:
            if 'DesignInfo' in json_document['DesignModule']:
                if 'DesignAllocation' in json_document['DesignModule']['DesignInfo']:
                    study_Type = json_document['DesignModule']['DesignInfo']['DesignAllocation']

                    for i, st in enumerate([study_Type]):
                        possed_source = getPOStags( str(st) )
                        studyType['name_'+str(i)] = possed_source

    return studyType 

def fetchStdTypeSources(json_document):

    combined_sources=dict()

    studyType = fetchStudyType(json_document)
    if studyType:
        combined_sources['s_type'] = studyType

    return combined_sources