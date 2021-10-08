

def generateMapping():

    mapping = dict()

    mapping['ep_sample_size'] =  ['BriefSummary', 'DetailedDescription']
    mapping['ep_gender'] =  ['BriefSummary', 'DetailedDescription']
    mapping['ep_age'] =  ['BriefTitle', 'OfficialTitle', 'BriefSummary', 'DetailedDescription']
    mapping['ep_condition'] =  ['BriefSummary', 'DetailedDescription']
    mapping['ei_name'] =  ['BriefSummary', 'DetailedDescription']
    mapping['ei_syn'] =  ['BriefSummary', 'DetailedDescription']
    mapping['es_type'] =  ['BriefSummary', 'DetailedDescription']

    return mapping