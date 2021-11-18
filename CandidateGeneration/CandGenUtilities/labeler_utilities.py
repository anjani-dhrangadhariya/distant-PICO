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
