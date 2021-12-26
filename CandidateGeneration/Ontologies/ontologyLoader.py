def OntologyLoaderDoc(a):
    '''The module loads all the ontologies and dictionaries that will be used for weak labeling in distant-PICO'''
    return a**a

print( OntologyLoaderDoc.__doc__ )

import csv
import sqlite3

import pandas as pd
import spacy

#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words
import string

additional_stopwords = ['of']
stopwords.update(additional_stopwords)

from Ontologies.OntoUtils import (allowedTermLength, countTerm, filterSAB,
                                  preprocessOntology, removeNonHuman)
from Ontologies.parseOntlogies import createMySQLConn



def selectTerminology(conn, pico_category):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """

    pico_category =  '%'+pico_category+'%'
    cur = conn.cursor()
    cur.execute("SELECT * FROM terminology1 WHERE PICO LIKE ?", (pico_category,))

    rows = cur.fetchall()

    return rows

'''
Description:
    This function loads the terms from the UMLS ontologies and groups them by Ontology and PICOS

Args:
    None 

Returns:
    UMLS ontologies (dict, dict, dict): three dictionaries (each corresponding to P, I/C and O) containing ontology terms grouped by Ontology 
'''
def loadUMLSdb(fpath, label, remove_vet = True):

    umls = dict()

    conn = createMySQLConn( fpath )

    rows = selectTerminology(conn, label)

    df = pd.DataFrame(rows, columns=['idx', 'SAB', 'TUI', 'CUI', 'TERM', 'STY', 'PICOS'])

    df['TERM_PRE'] = df.TERM.apply(preprocessOntology)

    df_new = df.groupby(['SAB']).apply(lambda x: list(zip( x.TERM_PRE , x.PICOS))).to_dict()

    if remove_vet == True:
        df_new = removeNonHuman(df_new)

    return df_new


'''
Description:
    This function loads Disease Ontolgoy DO terms and their synonyms

Args:
    None 

Returns:
    DO terms (list): A list containing all the terms and their synonyms from the Disease Ontology (DO)
'''
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

    return doid

'''
Description:
    This function loads Comparative Toxicogenomics Database (CTD) disease terms and their synonyms

Args:
    None 

Returns:
    CTD terms (list): A list containing all the terms corresponding to Disease set and their synonyms from the Comparative Toxicogenomics Database (CTD)
'''
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

    return ctd_disease

'''
Description:
    This function loads a dictionary of CDC (Centers for Disease Control and Prevention) Race and Ethnicity set compiled from
    https://www.cdc.gov/nchs/data/dvs/race_ethnicity_codeset.pdf

Args:
    None 

Returns:
    CDC dictionary (list): A list containing all the terms from CDC (Centers for Disease Control and Prevention) Race and Ethnicity set 
'''
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

'''
Description:
    This function loads the manually compiled dictionary of gender identities and sexual orientations
    sources: [https://www.hrc.org/resources/sexual-orientation-and-gender-identity-terminology-and-definitions, 
              https://www.itspronouncedmetrosexual.com/2013/01/a-comprehensive-list-of-lgbtq-term-definitions/,
              https://www.healthline.com/health/different-genders#takeaway]

Args:
    None 

Returns:
    Gender and sexual orientation terms (list): A list containing all the terms corresponding to gender identities and sexual orientations
'''
def loadGenders():

    inputFile = '/mnt/nas2/data/systematicReview/Ontologies/participant/gender_sexuality.txt'

    genders = []

    with open(inputFile, 'r') as fd:
        for line in fd:
            genders.append( line.strip() )

    return genders

'''
Description:
    This function loads Comparative Toxicogenomics Database (CTD) chemical terms and their synonyms

Args:
    None 

Returns:
    CTD terms (list): A list containing all the terms corresponding to Chemical set and their synonyms from the Comparative Toxicogenomics Database (CTD)
'''
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

    return ctd_chem


'''
Description:
    This function loads Chemical Entities of Biological Interest (ChEBI) terms and their synonyms

Args:
    None 

Returns:
    ChEBI terms (list): A list containing all the terms corresponding and their synonyms from the Chemical Entities of Biological Interest (ChEBI)
'''
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

    return chebi


def loadOntology():

    umls_p, umls_i, umls_o = loadUMLS()
    race_eth = loadRaceEthnicity()
    ctd_disease = loadCTDdisease()
    doid = loadDO()
    genders = loadGenders()

    ctd_chem = loadCTDchem()
    chebi = loadChEBI()

    return None
