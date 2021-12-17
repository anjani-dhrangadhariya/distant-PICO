import pandas as pd

def loadUMLS():

    inputFile = '/mnt/nas2/data/systematicReview/UMLS/english_subset/umls_preprocessed/concepts.tsv'

    umls_p = dict()
    umls_i = dict()
    umls_o = dict()

    umls_all = pd.read_csv(inputFile, sep='\t')



    return None


def loadDO():

    return None

def loadCTDdisease():

    return None

def loadRaceEthnicity():

    return None

def loadGenders():

    return None

def loadCTDchem():

    return None

def loadChEBI():

    return None


def loadOntology():

    return None


loadUMLS()