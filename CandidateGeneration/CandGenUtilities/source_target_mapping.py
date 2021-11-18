
'''
Description:
    This function generates mapping between the sources and targets sentences.

Args:
    None

Returns:
    dictionary: returns a "key: many values" dictionary mapping PICOS sources to appropriate targets
'''
def generateMapping():

    mapping = dict()

    mapping['ep_condition'] =  ['EligibilityCriteria', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'InterventionDescription', 'PriOutcomeDesc', 'SecOutcomeDesc']
    mapping['ep_age'] =  ['EligibilityCriteria', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'InterventionDescription', 'PriOutcomeDesc', 'SecOutcomeDesc']
    mapping['ep_gender'] =  ['EligibilityCriteria', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'InterventionDescription', 'PriOutcomeDesc', 'SecOutcomeDesc']
    mapping['ep_sample_size'] =  ['EligibilityCriteria', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'InterventionDescription', 'PriOutcomeDesc', 'SecOutcomeDesc']

    mapping['ei_name'] =  ['EligibilityCriteria', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'InterventionDescription', 'PriOutcomeDesc', 'SecOutcomeDesc']

    mapping['eo_name'] =  ['EligibilityCriteria', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'InterventionDescription', 'PriOutcomeDesc', 'SecOutcomeDesc']

    mapping['es_type'] =  ['EligibilityCriteria', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'InterventionDescription', 'PriOutcomeDesc', 'SecOutcomeDesc', 'DesignInterventionModelDescription']

    return mapping