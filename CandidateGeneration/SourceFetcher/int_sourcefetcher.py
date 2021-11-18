#!/usr/bin/env python

def IntSourceFetcherDoc(a):
    '''The module retrieves sources for each of the I - Intervention and C - Comparator entities'''
    return a**a

print( IntSourceFetcherDoc.__doc__ )

'''
Description:
    Extracts the dictionary of intervention information of the clinical trial from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    dictionary: a dictionary containing intervention mentions and arms group label mentions
'''
def fetchIntervention(protocol_section):

    interventionInfo = dict()

    if 'ArmsInterventionsModule' in protocol_section:
        if 'InterventionList' in protocol_section['ArmsInterventionsModule']:
            if 'Intervention' in protocol_section['ArmsInterventionsModule']['InterventionList']:
                intervention = protocol_section['ArmsInterventionsModule']['InterventionList']['Intervention']

                for i, eachIntervention in enumerate(intervention):
                    if 'InterventionName' in eachIntervention:
                        interventionInfo['name_' +str(i)] = eachIntervention['InterventionName']

                    if 'InterventionArmGroupLabelList' in eachIntervention:
                        interventionInfo['arm_name_' + str(i)] = eachIntervention['InterventionArmGroupLabelList']

    return interventionInfo

'''
Description:
    Fetches the synonym terms for the intervention (I) and comparator (C) subtype sources from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    dictionary: a dictionary containing synonyms for the intervention mentions and arms group label mentions
'''
def fetchInterventionSyn(protocol_section):
    interventionInfo = dict()

    if 'ArmsInterventionsModule' in protocol_section:
        if 'InterventionList' in protocol_section['ArmsInterventionsModule']:
            if 'Intervention' in protocol_section['ArmsInterventionsModule']['InterventionList']:
                intervention = protocol_section['ArmsInterventionsModule']['InterventionList']['Intervention']

                for i, eachIntervention in enumerate(intervention):
                    if 'InterventionOtherNameList' in eachIntervention:
                        synonym_list = eachIntervention['InterventionOtherNameList']['InterventionOtherName']
                        for j, eachSynonym in enumerate(synonym_list):
                            key = 'int_syn_' + str(i) + '_' + str(j)
                            interventionInfo[key] = eachSynonym

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

    # combined_sources['i_name'] = {**i_name, **i_syn}
    # combined_sources['i_name'] = i_name
    # combined_sources['i_synonym'] = i_syn

    combined_sources['i_name'] = {**i_name, **i_syn}

    return combined_sources