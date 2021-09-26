#!/usr/bin/env python

def PartiSourceFetcherDoc(a):
    '''The module retrieves sources for each of the P - participant entity'''
    return a**a

print( PartiSourceFetcherDoc.__doc__ )


'''
Description:
    Extracts the list of participant conditions for the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record

Returns:
    dictionary: a list of strings representing the participant condition names/terms ['', '', '']
'''
def fetchParticipantCondition(json_document):

    conditionList = []

    if 'ConditionsModule' in json_document:
        if 'ConditionList' in json_document['ConditionsModule']:
            conditionList = json_document['ConditionsModule']['ConditionList']['Condition']

    return conditionList

'''
Description:
    Extracts the string of gender/sex of the participants in the clinical trial from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record

Returns:
    string: a string representing the participant condition names/terms.  string ⊂ ['all', 'Male', 'Female']
'''
def fetchParticipantGender(json_document):

    if 'EligibilityModule' in json_document:
        if 'Gender' in json_document['EligibilityModule']:
            gender = json_document['EligibilityModule']['Gender']

    return gender

'''
Description:
    Extracts the dictionary of age information of the participants in the clinical trial from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    dictionary: a dictionary containing minimum and maximum participant ages and standardized MeSH age groups
'''
def fetchParticipantAge(json_document):

    stdAge = dict()

    if 'EligibilityModule' in json_document:
        if 'StdAgeList' in json_document['EligibilityModule']:
            stdAge = json_document['EligibilityModule']['StdAgeList']
        if 'MinimumAge' in json_document['EligibilityModule']:
            minAge = json_document['EligibilityModule']['MinimumAge']
            stdAge['MinimumAge'] = minAge
        if 'MaximumAge' in json_document['EligibilityModule']:
            maxAge = json_document['EligibilityModule']['MaximumAge']
            stdAge['MaximumAge'] = maxAge

    return stdAge # should be tuple of name

'''
Description:
    Extracts the string of sample size information of the participants in the clinical trial from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    string: a string containing sample size information in the clinical trials
'''
def fetchParticipantSampSize(json_document):

    if 'DesignModule' in json_document:
            if 'EnrollmentInfo' in json_document['DesignModule']:
                sampleSize = json_document['DesignModule']['EnrollmentInfo']['EnrollmentCount']

    return sampleSize 


'''
Description:
    Fetches participant (P) subtype sources from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    dictionary: a dictionary containing all the participant subtype sources
'''
def fetchParticipantSources(json_document):

    combined_sources = dict()

    p_condition = fetchParticipantCondition(json_document)
    p_age = fetchParticipantAge(json_document)
    p_gender = fetchParticipantGender(json_document)
    p_sampsize = fetchParticipantSampSize(json_document)

    combined_sources['p_condition'] = p_condition
    combined_sources['p_age'] = p_age
    combined_sources['p_gender'] = p_gender
    combined_sources['p_sample_size'] = p_sampsize

    return combined_sources