#!/usr/bin/env python

def IntSourceFetcherDoc(a):
    '''The module retrieves sources for each of the I - Intervention and C - Comparator entities'''
    return a**a

print( IntSourceFetcherDoc.__doc__ )

'''
Description:
    Extracts the dictionary of intervention information in the clinical trial from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    dictionary: a dictionary containing intervention information in the clinical trials
'''
def fetchIntervention(protocol_section):

    interventionInfo = dict()

    if 'ArmsInterventionsModule' in protocol_section:
        if 'InterventionList' in protocol_section['ArmsInterventionsModule']:
            if 'Intervention' in protocol_section['ArmsInterventionsModule']['InterventionList']:
                intervention = protocol_section['ArmsInterventionsModule']['InterventionList']['Intervention']

                for eachIntervention in intervention:
                    if 'InterventionName' in eachIntervention:
                        interventionInfo['name'] = eachIntervention['InterventionName']

                    if 'InterventionArmGroupLabelList' in eachIntervention:
                        interventionInfo['arm_name'] = eachIntervention['InterventionArmGroupLabelList']

    return interventionInfo

'''
Description:
    Fetches the synonym terms for the intervention (I) and comparator (C) subtype sources from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    dictionary: a dictionary containing all the synonyms for the intervention and comparator subtype sources
'''
def fetchInterventionSyn(protocol_section):
    interventionInfo = dict()

    if 'ArmsInterventionsModule' in protocol_section:
        if 'InterventionList' in protocol_section['ArmsInterventionsModule']:
            if 'Intervention' in protocol_section['ArmsInterventionsModule']['InterventionList']:
                intervention = protocol_section['ArmsInterventionsModule']['InterventionList']['Intervention']

                for eachIntervention in intervention:
                    if 'InterventionOtherNameList' in eachIntervention:
                        interventionInfo['int_syn'] = eachIntervention['InterventionOtherNameList']['InterventionOtherName']

    return interventionInfo


'''
Description:
    Fetches intervention (I) and comparator (C) subtype sources from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    dictionary: a dictionary containing all the intervention and comparator subtype sources
'''
def fetchIntcompSources(json_document):

    combined_sources = dict()

    i_name = fetchIntervention(json_document)
    i_syn = fetchInterventionSyn(json_document)

    combined_sources['i_name'] = i_name
    combined_sources['i_synonym'] = i_syn

    return combined_sources