def allowedPOS():
    return list(['NOUN', 'PROPN', 'VERB', 'ADJ'])

def generateLabels():

    PICOS = dict()

    PICOS['p'] = 1
    PICOS['i'] = 2
    PICOS['o'] = 3
    PICOS['s'] = 4
    PICOS['ABSTAIN'] = -1
    PICOS['Ospan'] = 0
    
    return PICOS

def generateAntiLabels(PICOS):
    return {v: k for k, v in PICOS.items()}


def abstainOption():

    abstain = dict()

    abstain['p_condition'] = True
    abstain['p_age'] = False
    abstain['p_gender'] = False
    abstain['p_sample_size'] = False
    abstain['i_name'] = True
    abstain['o_name'] = True
    abstain['s_type'] = False

    return abstain


def punct():
    return ['-', '/', '\\']