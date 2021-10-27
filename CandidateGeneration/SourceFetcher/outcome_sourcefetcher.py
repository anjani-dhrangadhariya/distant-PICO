#!/usr/bin/env python

def OutSourceFetcherDoc(a):
    '''The module retrieves sources for each of the O - Outcome entity'''
    return a**a

print( OutSourceFetcherDoc.__doc__ )

'''
Description:
    Extracts the list of strings containing primary outcomes information in the clinical trial from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    dictionary: a dictionary containing primary outcome information in the clinical trials
'''
def fetchPrimaryOutcome(json_document):

    outcomes_list = dict()

    if 'OutcomesModule' in json_document:
            if 'PrimaryOutcomeList' in json_document['OutcomesModule']:
                primOutcome = json_document['OutcomesModule']['PrimaryOutcomeList']['PrimaryOutcome']
                for i, eachOutcome in enumerate(primOutcome):
                    if 'PrimaryOutcomeMeasure' in eachOutcome:
                        primOutcome = eachOutcome['PrimaryOutcomeMeasure']
                        outcomes_list['PrimaryOutcome_' + str(i)] = primOutcome

    return outcomes_list

'''
Description:
    Extracts the list of strings containing secondary outcomes information in the clinical trial from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    dictionary: a dictionary containing secondary outcome information in the clinical trials
'''
def fetchSecondaryOutcome(json_document):

    outcomes_list = dict()

    if 'OutcomesModule' in json_document:
            if 'SecondaryOutcomeList' in json_document['OutcomesModule']:
                secondOutcome = json_document['OutcomesModule']['SecondaryOutcomeList']['SecondaryOutcome']
                for i, eachOutcome in enumerate(secondOutcome) :
                    if 'SecondaryOutcomeMeasure' in eachOutcome:
                        secondOutcome = eachOutcome['SecondaryOutcomeMeasure']
                        outcomes_list['SecondaryOutcome_'+str(i)] = secondOutcome

    return outcomes_list 

'''
Description:
    Fetches outcome (O) subtype sources from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    dictionary: a dictionary containing all the outcome subtype sources
'''
def fetchOutcomeSources(json_document):

    combined_sources = dict()

    o_primary = fetchOutcome(json_document)
    combined_sources['o_primary'] = o_primary

    o_secondary = fetchSecondaryOutcome(json_document)
    combined_sources['o_secondary'] = o_secondary

    return combined_sources