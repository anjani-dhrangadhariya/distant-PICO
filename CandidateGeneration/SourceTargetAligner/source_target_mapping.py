

def generateMapping():

    mapping = dict()

    mapping['ep_sample_size'] =  [ 'BriefSummary', 'DetailedDescription']
    mapping['ep_gender'] =  ['BriefSummary', 'DetailedDescription', 'PriOutcomeDesc', 'SecOutcomeDesc']
    mapping['ep_age'] =  ['EligibilityCriteria', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription']
    mapping['ep_condition'] =  ['EligibilityCriteria', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription']

    mapping['ei_name'] =  ['BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'InterventionDescription']
    mapping['ei_syn'] =  ['BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'InterventionDescription']

    mapping['eo_primary'] =  ['PriOutcomeDesc', 'SecOutcomeDesc']
    mapping['eo_secondary'] =  ['PriOutcomeDesc', 'SecOutcomeDesc']

    mapping['es_type'] =  ['InterventionDescription', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'DesignInterventionModelDescription']

    return mapping