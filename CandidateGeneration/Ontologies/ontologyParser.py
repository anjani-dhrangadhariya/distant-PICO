#!/usr/bin/env python

def ParseOntDoc(a):
    '''The module maps CUIs to TUIs from the selected UMLS subset'''
    return a**a

print( ParseOntDoc.__doc__ )

# Generic imports
import csv
import errno
import os
import sqlite3
import string
from collections import defaultdict
from sqlite3 import Error
# import Ontologies

# Data science generic libraries
import numpy as np
import pandas as pd
import msgpack

# NLP-specifc imports
import spacy
from scispacy.abbreviation import AbbreviationDetector

import spacy
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

en = spacy.load('en_core_web_sm')
en.add_pipe("abbreviation_detector")
#stopwords = en.Defaults.stop_words

def loadStopWords():

    numbers_list = ['sixty', 'fifteen', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'first', 'second', 'third', 'fourth', 'hundred', 'twenty']

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

    # Remove numbers from the stopwords list
    stopwords_lf = [ sw for sw in stopwords_lf if sw not in numbers_list]

    return stopwords_lf

general_stopwords = loadStopWords()

'''
Description:
    This function could be used for preprocessing ontology terms. Preprocessing includes 1) Remove stopwords 2) Remove numbers 3) Remove punctuations

Args:
    term (str): String variable containing the ontology term

Returns:
    preprocessed term (str): String variable containing the preprocessed ontology term
'''
def preprocessOntology(term):

    # remove stopwords
    lst = [ token for token in term.split() if token.lower() not in general_stopwords ]
    lst = ' '.join(lst)

    # remove numbers
    numRemove = ''.join([i for i in lst if not i.isdigit()])

    # remove punctuation
    punctRemove = numRemove.translate(str.maketrans(' ', ' ', string.punctuation))

    return punctRemove

'''
Description:
    The function fetches POS tags for the input string

Args:
    String: free-text string

Returns:
    Dictionary (dict): returns a dictionary containing free-text string with its tokenized string, token lemma, POS tags for the tokens, finer POS tags for the token
'''
def getPOStags(value):

    doc = en(value)

    tokens = [ token.text for token in doc ]
    pos_fine = [ token.tag_ for token in doc ]

    return pd.Series( { 'TOKENS':tokens, 'POS_FINE':pos_fine } )

'''
Description:
    Creates a database connection to the SQLite database specified by the db_file

Args:
    db_file (str): Path to the SQLite database 

Returns:
    conn (str): Connection object or None
'''
def createMySQLConn(db_file):

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print('This is the connection: ', conn)
    except Error as e:
        print(e)

    return conn

'''
Description:
    Creates a database connection to the SQLite database specified by the fpath and load UMLS data into the database from dataframe

Args:
    fpath (str): Path to the SQLite database 
    dataframe (DataFrame): Dataframe containing UMLS data to load into the SQLite database passed through fpath

Returns:
    
'''
def init_sqlite_tables(fpath, dataframe):

    conn = createMySQLConn(fpath)
    sql = """CREATE TABLE IF NOT EXISTS terminology1 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    SAB text NOT NULL,
                    TUI text NOT NULL,
                    CUI text NOT NULL,
                    TERM text NOT NULL,
                    STY text NOT NULL,
                    P text NOT NULL,
                    I text NOT NULL,
                    O text NOT NULL,
                    S text NOT NULL,
                    TERM_PRE text NOT NULL                  
                );"""
    conn.execute(sql)

    sql = """CREATE INDEX IF NOT EXISTS idx_sources 
                        ON terminology1 (sab);"""
    conn.execute(sql)

    sql = """CREATE INDEX IF NOT EXISTS idx_source_terms 
                ON terminology1 (sab, tui);"""
    conn.execute(sql)

    rows = list(dataframe.itertuples())
    conn.executemany(
        "INSERT into terminology1(SAB, TUI, CUI, TERM, STY, P, I, O, S, TERM_PRE) values (?,?,?,?,?,?,?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()

'''
Description:
    Maps UMLS CUI to TUI, writes them to a file specified by tui2pio, preprocesses the ontologies, generates POS tags and writes them to a UMLS
    database using function init_sqlite_tables

Args:
    indir (str): Path to input directory with raw UMLS files 
    outdir (str): TODO
    tui2pio (str):  TODO

Returns:
    
'''
def cui2tuiMapper(indir, outdir, tui2pio):

    # validate that the UMLS source REFs are provided
    for fname in ['MRCONSO.RRF', 'MRSTY.RRF', 'MRSAB.RRF']:
        if not os.path.exists(f"{indir}/{fname}"):
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                fname
            )

    # Source terminologies - MRSAB.RRF
    sabs = {}
    with open(f'{indir}/MRSAB.RRF', 'r') as fp:
        for line in fp:
            row = line.strip('').split('|')
            # ignore RSAB version
            rsab, _, lat, ssn = row[3], row[6], row[19], row[23]
            if rsab in sabs:
                continue
            sabs[rsab] = (rsab, lat, ssn)

    with open(f'{outdir}/sabs.bin', 'wb') as fp:
        fp.write(msgpack.dumps(sabs))

    #Concept Unique ID to Semantic Type mappings - MRSTY.RRF
    tui_to_sty = {}
    cui_to_tui = defaultdict(set)
    tui2pio_mapping = dict()

    with open(f'{indir}/MRSTY.RRF', 'r') as fp, open(tui2pio, 'r') as pio_fp:

        rd = csv.reader(pio_fp, delimiter="\t", quotechar='"')
        for row in rd:
            tui2pio_mapping[row[0]] = [ row[2] , row[3], row[4], row[5] ]

        for line in fp:
            row = line.strip('').split('|')
            cui, tui, sty = row[0], row[1], row[3]
            cui_to_tui[cui].add(tui)
            tui_to_sty[tui] = sty
            tui_to_sty[tui] = sty

    with open(f'{outdir}/tui_to_sty.bin', 'wb') as fp:
        fp.write(msgpack.dumps(tui_to_sty))

    # MRCONSO.RRF
    with open(f'{indir}/MRCONSO.RRF', 'r') as fp, open(
            f'{outdir}/concepts.tsv', 'w') as op:
        op.write('SAB\tTUI\tCUI\tTERM\tSTY\tP\tI\tO\tS\n')
        for line in fp:
            row = line.strip().split('|')
            cui, sab, term = row[0], row[11], row[14]
            if term.strip() is None:
                continue
            for tui in cui_to_tui[cui]:
                val = tui2pio_mapping[tui]
                op.write(f'{sab}\t{tui}\t{cui}\t{term}\t{tui_to_sty[tui]}\t{val[0]}\t{val[1]}\t{val[2]}\t{val[3]}\n')


    print('Completed mapping CUIs to TUIs and stored the file in directory: ', outdir)
    

    df = pd.read_csv(
        f'{outdir}/concepts.tsv',
        sep='\t',
        header=0,
        quotechar=None,
        quoting=3,
        index_col=0,
        na_filter=False,
        dtype={
            'SAB': 'object',
            'TUI': 'object',
            'CUI': 'object',
            'TERM': 'object',
            'STY': 'object',
            'P': 'object',
            'I': 'object',
            'O': 'object',
            'S': 'object'
        }
    )

    # Preprocess the ontology terms here
    df['TERM_PRE'] = df.TERM.apply(preprocessOntology)

    # Add POS-tags
    # df = df.merge( df.TERM_PRE.apply(getPOStags) , left_index=True, right_index=True )

    # Open the written file and load it into MySQL
    init_sqlite_tables(f'{outdir}/umls_tui_pios4_.db', df)

indir = '/mnt/nas2/data/systematicReview/UMLS/english_subset/2021AB/META'
outdir = '/mnt/nas2/data/systematicReview/UMLS/english_subset/umls_preprocessed'
f_tui2pio = '/mnt/nas2/data/systematicReview/UMLS/english_subset/umls_preprocessed/tui_pio_v4.tsv'

# cui2tuiMapper(indir, outdir, f_tui2pio)