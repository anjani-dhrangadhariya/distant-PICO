def OntologyLoaderDoc(a):
    '''The module loads all the ontologies and dictionaries that will be used for weak labeling in distant-PICO'''
    return a**a

print( OntologyLoaderDoc.__doc__ )

import csv
import json
import re
import sqlite3
import string
from pathlib import Path

import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')


translator = str.maketrans(' ', ' ', string.punctuation)

from Ontologies.ontologyParser import createMySQLConn
from Ontologies.ontoUtils import (allowedTermLength, countTerm, filterSAB,
                                  preprocessOntology, removeNonHuman,
                                  removeTerms, termCountThreshold)

'''
Description:
    This function selects all rows from UMLS.db for the chosen pico_category.

Args:
    conn (Connection) : MySQL connection to the preprocessed UMLS.db 
    pico_category (str): choice for the entitiy type to retrieve appropriate UMLS concepts from UMLS.db

Returns:
    rows (list): Selected terms (and their SAB label) for  pico_category
'''
def selectTerminology(conn, pico_category):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """

    pico_category_pattern =  '%'+pico_category+'%'
    cur = conn.cursor()
    cur.execute("SELECT * FROM terminology1 WHERE ? LIKE ?", (pico_category, pico_category_pattern,))
    # cur.execute("SELECT * FROM terminology1 WHERE ? LIKE ? LIMIT 50000", (pico_category, pico_category_pattern,))

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

    df = pd.DataFrame(rows, columns=['idx', 'SAB', 'TUI', 'CUI', 'TERM', 'STY', 'P', 'I', 'O', 'TERM_PRE'])

    # df['TERM_PRE'] = df.TERM.apply(preprocessOntology)

    if entity == 'P':
        df_new = df.groupby(['SAB']).apply(lambda x: list(zip( x.TERM_PRE , x.P))).to_dict()
    if entity == 'I':
        df_new = df.groupby(['SAB']).apply(lambda x: list(zip( x.TERM_PRE , x.I))).to_dict()
    if entity == 'O':
        df_new = df.groupby(['SAB']).apply(lambda x: list(zip( x.TERM_PRE , x.O))).to_dict()    

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
def loadOnt(fpath:str, delim:str, term_index:int, term_syn_index:int, char_threshold:int = 2):

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
    fpath (str): path to the input hand-crafted dictionary file

Returns:
    Gender and sexual orientation terms (list): A list containing all the terms corresponding to gender identities and sexual orientations
'''
def loadDict(fpath:str):

    terms = []

    with open(fpath, 'r') as fd:
        for line in fd:
            # preprocessing
            line_preprocessed = line.translate( translator )
            terms.append( line_preprocessed.strip() )

    return terms

'''
Description:
    Loads abbreviation source files from local directory into a list 
Args:
    fpath (str): path to the input abbreviation file

Returns:
    abbs (list): The list containing abbreviations from the input file
'''
def loadAbbreviations(fpath:str):

    abbs = []

    with open(fpath, 'r') as fd:
        for line in fd:
            abb = line.split('\t')[0]
            abbs.append( abb.strip() )

    return abbs

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
def loadDS(fpath:str, picos:str, char_threshold:int = 2):

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
def loadPattern( pattern_name:str ):

    if pattern_name == 'samplesize':

        samp_size_pattern =  r'([0-9]+ ?([a-zA-Z0-9]+)? ?(patients?|subjects?|participants?|people?|individuals?|persons?|healthy individuals?|healthy adults?|children|toddlers?|adults?|healthy volunteers?|families?|men|women|teenagers?|families|parturients?|females?|males?)+)'
        compiled_pattern = re.compile(samp_size_pattern)
        return compiled_pattern

    if pattern_name == 'samplesize2': # Sample size in ReGeX expression (n=XXXX)

        samp_size_pattern =  r'(\(\s?n\s?=\s?[0-9]+\s?\))'
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

def loadExternalModel(fpath:str ):

    # TODO: Loads an external model (.pth) from a path onto CUDA

    return None

def loadStopWords():

    stopwords_lf = []

    # NLTK
    nltk_stopwords = list(stopwords.words('english'))
    # print( 'Total number of stopwords in NLTK: ', len( nltk_stopwords ) )
    stopwords_lf.extend( nltk_stopwords )

    # gensim
    # print( 'Total number of stopwords in Gensim: ', len( STOPWORDS ) )
    stopwords_lf.extend( STOPWORDS )

    # scikit learn
    # print( 'Total number of stopwords in scikit learn: ', len( ENGLISH_STOP_WORDS ) )
    stopwords_lf.extend( ENGLISH_STOP_WORDS )

    # spacy
    spacy_stopwords = en.Defaults.stop_words
    # print( 'Total number of stopwords in Spacy: ', len( spacy_stopwords ) )
    stopwords_lf.extend( spacy_stopwords )

    # additional stopwords
    additional_stopwords = ['of']
    stopwords_lf.extend(additional_stopwords)

    return list( set(stopwords_lf) ) 