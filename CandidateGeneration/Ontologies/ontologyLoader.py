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
    This function loads and returns concepts from the preprocessed local dump of UMLS database and groups them by Ontology (SAB) and entity type

Args:
    fPath : File path to the preprocessed UMLS.db 
    entity: choice for the entitiy type to retrieve appropriate UMLS concepts from
    remove_vet : setting it True will remove the concepts from veterinary terminologies. 
    min_terms : setting it will remove a terminology (SAB) if #concepts in an SAB < min_terms
    char_threshold: setting it will remove all concepts from SAB shoter than char_threshold

Returns:
    UMLS ontologies (dict): the dictionary containing concepts grouped by Ontology (SAB) for chosen entity
'''
def loadUMLSdb(fpath, entity: str, remove_vet: bool = True, min_terms: int = 500, char_threshold:int = 3):

    umls = dict()

    conn = createMySQLConn( fpath )

    rows = selectTerminology(conn, entity)

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
    Loads ontology files from local directory. 
Args:
    fpath (str): path to the input ontology file
    delim (str): delimiter for the input ontology file (tab for TSV and comma for CSV)
    term_index (int): csv/tsv file index to retrieve term
    term_syn_index (int): csv/tsv file index to retrieve term synonym
    char_threshold (int): setting it will remove all concepts from SAB shoter than char_threshold

Returns:
    term_prepro (list): The list containing all the preprocessed terms for the input ontology
    term_syn_prepro (list): The list containing all the preprocessed term synonym for the input ontology

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

'''
Description:
    Loads distant supervision source files from local directory. 
Args:
    fpath (str): path to the distant supervision directory
    picos (str): part of file path for a particular (P/IC/O) distant supervision source
    char_threshold (int): setting it will remove all concepts from SAB shoter than char_threshold

Returns:
    ds_source_prepro (list): The list containing all the preprocessed distant supervision terms from clinicaltrials.org
'''
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

'''
Description:
    Loads chosen ReGeX pattern 
Args:
    pattern_name (str): pattern name 

Returns:
    compiled_pattern (re): compiled ReGeX pattern for the chosen pattern_name
'''
def loadPattern( pattern_name ):

    if pattern_name == 'samplesize':

        # samp_size_pattern =  r'([0-9]+ ?(patients?|subjects?|participants?|people?|individuals?|persons?|healthy individuals?|healthy adults?|children?|toddlers?|adult?adults?|healthy volunteers?|families?|men?|women?|teenagers?|families?|parturients?)+)'
        samp_size_pattern =  r'([0-9]+ ?([a-zA-Z0-9]+)? ?(patients?|subjects?|participants?|people?|individuals?|persons?|healthy individuals?|healthy adults?|children|toddlers?|adults?|healthy volunteers?|families?|men|women|teenagers?|families|parturients?|females?|males?)+)'
        compiled_pattern = re.compile(samp_size_pattern)
        return compiled_pattern

    if pattern_name == 'samplesize2': # Sample size in ReGeX expression (n=XXXX)

        samp_size_pattern =  r'\( n = [0-9,]+ \)?'
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

    if pattern_name == 'meanage':

        mean_age_pattern = r'(([Mm]ean|[Aa]verage) ?(age) ?(=|was|of|,|:|) ?[0-9.± ()+-\/]+(years?|months?|yr?s?)?)'
        compiled_pattern = re.compile(mean_age_pattern)
        return compiled_pattern

    if pattern_name == 'studytype':

        mean_age_pattern = r'(([Nn]o[nt] )?([rR]andom(i[sz]ed|ly|i[sz]ation)?)+(,? controlled| clinical)?( trials?)?)'
        compiled_pattern = re.compile(mean_age_pattern)
        return compiled_pattern

def loadExternalModel(fpath):

    # TODO: Loads a model from a path onto CUDA

    return None