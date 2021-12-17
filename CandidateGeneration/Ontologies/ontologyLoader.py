import pandas as pd
import csv

def preprocessOntology():

    return None

def loadUMLS():

    inputFile = '/mnt/nas2/data/systematicReview/UMLS/english_subset/umls_preprocessed/concepts.tsv'

    umls_p = dict()
    umls_i = dict()
    umls_o = dict()

    # Calculate term coverage on the training targets
    # Term coverage = summation over entire terminology for term frequency in document (each document is each NCTID but we should consider the entire corpus here)
    with open(inputFile) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        next(rd, None)
        for counter, row in enumerate(rd):
            ontology = row[0]
            term = row[3]

            if 'P' in row[-1] and '-' not in row[-1]:
                if ontology not in umls_p:
                    umls_p[ ontology ] = [term]
                else:
                    umls_p[ontology].append(term)

            if 'I' in row[-1] and '-' not in row[-1]:
                if ontology not in umls_i:
                    umls_i[ ontology ] = [term]
                else:
                    umls_i[ontology].append(term)

            if 'O' in row[-1] and '-' not in row[-1]:
                if ontology not in umls_o:
                    umls_o[ ontology ] = [term]
                else:
                    umls_o[ontology].append(term)

    print(len(umls_p))
    print(len(umls_i))
    print(len(umls_o))

    return None


def loadDO():

    inputFile = '/mnt/nas2/data/systematicReview/Ontologies/participant/DOID.csv'
    doid = []

    with open(inputFile) as fd:
        rd = csv.reader(fd, delimiter=",")
        next(rd, None)
        for counter, row in enumerate(rd):
            doid.append( row[1] )
            if row[2]:
                synonyms = row[2]
                synonyms = synonyms.split('|')
                doid.extend( synonyms )

    return None

def loadCTDdisease():

    inputFile = '/mnt/nas2/data/systematicReview/Ontologies/participant/CTD_diseases.csv'
    ctd_disease = []

    with open(inputFile) as fd:
        rd = csv.reader(fd, delimiter=",")
        next(rd, None)
        for counter, row in enumerate(rd):
            ctd_disease.append( row[0] )
            if row[7]:
                synonyms = row[7]
                synonyms = synonyms.split('|')
                ctd_disease.extend( synonyms )

    return None

def loadRaceEthnicity():

    inputFile = '/mnt/nas2/data/systematicReview/Ontologies/participant/cdc_race_ethnicity_codeset_v1.csv'

    race_eth = []

    with open(inputFile) as fd:
        rd = csv.reader(fd, delimiter=",")
        next(rd, None)
        for counter, row in enumerate(rd):
            race_eth.append( row[2] )
            if row[3]:
                race_eth.append( row[3] )

    return race_eth

def loadGenders():

    return None

def loadCTDchem():

    inputFile = '/mnt/nas2/data/systematicReview/Ontologies/intervention/CTD_chemicals.tsv'
    ctd_chem = []

    with open(inputFile) as fd:
        rd = csv.reader(fd, delimiter="\t")
        next(rd, None)
        for counter, row in enumerate(rd):
            ctd_chem.append( row[0] )
            if len(row[7]) > 2:
                synonyms = row[7]
                synonyms = synonyms.split('|')
                ctd_chem.extend( synonyms )

    return None

def loadChEBI():

    inputFile = '/mnt/nas2/data/systematicReview/Ontologies/intervention/CHEBI.csv'

    chebi = []

    with open(inputFile) as fd:
        rd = csv.reader(fd, delimiter=",")
        next(rd, None)
        for counter, row in enumerate(rd):
            chebi.append( row[1] )
            if row[2]:
                synonyms = row[2]
                synonyms = synonyms.split('|')
                chebi.extend( synonyms )

    return None


def loadOntology():

    loadUMLS()
    loadRaceEthnicity()
    loadCTDdisease()
    loadDO()
    loadGenders()

    loadCTDchem()
    loadChEBI()

    return None


loadOntology()