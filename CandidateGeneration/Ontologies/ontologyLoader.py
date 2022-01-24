def OntologyLoaderDoc(a):
    '''The module loads all the ontologies and dictionaries that will be used for weak labeling in distant-PICO'''
    return a**a

print( OntologyLoaderDoc.__doc__ )

import csv
import json
import sqlite3
import string
from pathlib import Path

import pandas as pd
import spacy

import re

#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words


additional_stopwords = ['of']
stopwords.update(additional_stopwords)

from Ontologies.ontoUtils import (allowedTermLength, countTerm, filterSAB,
                                  preprocessOntology, removeNonHuman,
                                  removeTerms, termCountThreshold)
from Ontologies.ontologyParser import createMySQLConn


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
def loadUMLSdb(fpath, label, remove_vet: bool = True, min_terms: int = 500, char_threshold:int = 3):

    umls = dict()

    conn = createMySQLConn( fpath )

    rows = selectTerminology(conn, label)

    df = pd.DataFrame(rows, columns=['idx', 'SAB', 'TUI', 'CUI', 'TERM', 'STY', 'PICOS', 'TERM_PRE'])

    # df['TERM_PRE'] = df.TERM.apply(preprocessOntology)

    df_new = df.groupby(['SAB']).apply(lambda x: list(zip( x.TERM_PRE , x.PICOS))).to_dict()

    if remove_vet == True:
        df_new = removeNonHuman(df_new)

    # Remove terms with less than 'char_threshold characters
    if char_threshold:
        df_new = removeTerms( df_new, char_threshold )

    # Remove the SAB with less than X terms
    if min_terms:
        df_new = termCountThreshold( df_new )

    return df_new


'''
Description:
    This function loads Comparative Toxicogenomics Database (CTD) disease terms and their synonyms
    This function loads Comparative Toxicogenomics Database (CTD) chemical terms and their synonyms
    This function loads Chemical Entities of Biological Interest (ChEBI) terms and their synonyms
    This function loads Disease Ontolgoy DO terms and their synonyms

Args:
    fpath, delim, term_index, term_syn_index

Returns:
    CTD terms (list): A list containing all the terms corresponding to Disease set and their synonyms from the Comparative Toxicogenomics Database (CTD)
    CTD terms (list): A list containing all the terms corresponding to Chemical set and their synonyms from the Comparative Toxicogenomics Database (CTD)
    ChEBI terms (list): A list containing all the terms corresponding and their synonyms from the Chemical Entities of Biological Interest (ChEBI)
    DO terms (list): A list containing all the terms and their synonyms from the Disease Ontology (DO)
'''
def loadOnt(fpath, delim, term_index, term_syn_index, char_threshold:int = 2):

    term = []
    term_syn = []

    with open(fpath) as fd:
        rd = csv.reader(fd, delimiter = delim)
        next(rd, None)
        for counter, row in enumerate(rd):
            if len( row[term_index] ) > char_threshold:
                term.append( row[term_index] )
            if row[term_syn_index]:
                synonyms = row[term_syn_index]
                synonyms = synonyms.split('|')
                synonyms = [s for s in synonyms if len(s) > char_threshold]
                term_syn.extend( synonyms )
    
    term_prepro = list(map(preprocessOntology, term))
    term_syn_prepro = list(map(preprocessOntology, term_syn))

    return term_prepro, term_syn_prepro

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
def loadDict(fpath):

    terms = []

    with open(fpath, 'r') as fd:
        for line in fd:
            terms.append( line.strip() )

    return terms

def loadAbbreviations(fpath):

    terms = []

    with open(fpath, 'r') as fd:
        for line in fd:
            abb = line.split('\t')[0]
            terms.append( abb.strip() )

    return terms

def loadDS(fpath, picos, char_threshold:int = 2):

    ds_source = []

    with open(f'{fpath}/{picos}.txt', 'r') as fp:

        for sourceLine in fp:
            sourceJSON = json.loads(sourceLine)

            for k,v in sourceJSON.items():
                if len( v['text'] ) > char_threshold:
                    ds_source.append( v['text'] )

    ds_source = list( set( ds_source ) )
    ds_source_prepro = list(map(preprocessOntology, ds_source))

    return ds_source_prepro

def loadPattern( pattern_name ):

    if pattern_name == 'samplesize':

        samp_size_pattern =  r'([0-9]+ ?(patients?|subjects?|participants?|people?|individuals?|persons?|healthy individuals?|healthy adults?|children?|toddlers?adult?adults?|healthy volunteers?|families?|men?|women?|teenagers?|families?|parturients?)+)'
        compiled_pattern = re.compile(samp_size_pattern)
        return compiled_pattern

    if pattern_name == 'age1':

        age_range_pattern =  r'(([Aa]ge[ds]? )?\b([0-9]{1,2})\b(\s+years?|\s+months?)?(\s+old|-old)?\s?(-|to)\s?\b([0-9]{1,2})\b(\s+years?|\s+months?)+(\s+old|-old)?)'
        compiled_pattern = re.compile(age_range_pattern)
        return compiled_pattern

    if pattern_name == 'age2':

        age_range_pattern =  r'(([Aa]ge[ds]? ?)\b([0-9]{1,2})\b(\s+years?|\s+months?)?(\s+old|-old)?\s?(and above| and older)?)'
        compiled_pattern = re.compile(age_range_pattern)
        return compiled_pattern

def loadExternalModel(fpath):

    # Loads a model from a path onto CUDA
    

    return None