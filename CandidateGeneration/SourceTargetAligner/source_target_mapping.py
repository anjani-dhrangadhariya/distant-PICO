
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

    mapping['ep_condition'] =  ['EligibilityCriteria', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription']
    mapping['ep_age'] =  ['EligibilityCriteria', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription']
    mapping['ep_gender'] =  ['EligibilityCriteria', 'BriefSummary', 'DetailedDescription', 'PriOutcomeDesc', 'SecOutcomeDesc']
    mapping['ep_sample_size'] =  ['BriefSummary', 'DetailedDescription']


    mapping['ei_name'] =  ['BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'InterventionDescription']
    mapping['ei_syn'] =  ['BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'InterventionDescription']


    mapping['eo_name'] =  ['PriOutcomeDesc', 'SecOutcomeDesc', 'BriefSummary', 'DetailedDescription', 'BriefTitle', 'OfficialTitle']
    # mapping['eo_secondary'] =  ['PriOutcomeDesc', 'SecOutcomeDesc', 'BriefSummary', 'DetailedDescription', 'BriefTitle', 'OfficialTitle']


    mapping['es_type'] =  ['InterventionDescription', 'BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription', 'DesignInterventionModelDescription']

    return mapping


def generateLabels():

    PICOS = dict()

    PICOS['P'] = 1
    PICOS['IC'] = 2
    PICOS['O'] = 3
    PICOS['S'] = 4
    
    return PICOS

def generateAntiLabels(PICOS):
    return {v: k for k, v in PICOS.items()}
