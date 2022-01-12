#!/usr/bin/env python

def ParseOntDoc(a):
    '''The module maps CUIs to TUIs from the selected UMLS subset'''
    return a**a

print( ParseOntDoc.__doc__ )

import csv
import errno
import os
import sqlite3
from collections import defaultdict
from sqlite3 import Error

import msgpack
import pandas as pd
import spacy

#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words
import string

additional_stopwords = ['of']
stopwords.update(additional_stopwords)

def preprocessOntology(term):

    # remove stopwords
    lst = [ token for token in term.split() if token.lower() not in stopwords ]
    lst = ' '.join(lst)

    # remove numbers
    numRemove = ''.join([i for i in lst if not i.isdigit()])

    # remove punctuation
    punctRemove = numRemove.translate(str.maketrans(' ', ' ', string.punctuation))

    return punctRemove

def createMySQLConn(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

# @staticmethod
def init_sqlite_tables(fpath, dataframe):

    conn = createMySQLConn(fpath)
    sql = """CREATE TABLE IF NOT EXISTS terminology1 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    SAB text NOT NULL,
                    TUI text NOT NULL,
                    CUI text NOT NULL,
                    TERM text NOT NULL,
                    STY text NOT NULL,
                    PICO text NOT NULL,
                    TERM_PRE text not NULL
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
        "INSERT into terminology1(SAB, TUI, CUI, TERM, STY, PICO, TERM_PRE) values (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

# @staticmethod
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
            tui2pio_mapping[row[0]] = row[2]

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
        op.write('SAB\tTUI\tCUI\tTERM\tSTY\tPICO\n')
        for line in fp:
            row = line.strip().split('|')
            cui, sab, term = row[0], row[11], row[14]
            if term.strip() is None:
                continue
            for tui in cui_to_tui[cui]:
                op.write(f'{sab}\t{tui}\t{cui}\t{term}\t{tui_to_sty[tui]}\t{tui2pio_mapping[tui]}\n')


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
            'PICO': 'object'
        }
    )

    # Preprocess the ontology terms here
    df['TERM_PRE'] = df.TERM.apply(preprocessOntology)

    # Open the written file and load it into MySQL
    init_sqlite_tables(f'{outdir}/umls_pre.db', df)

indir = '/mnt/nas2/data/systematicReview/UMLS/english_subset/2021AB/META'
outdir = '/mnt/nas2/data/systematicReview/UMLS/english_subset/umls_preprocessed'
f_tui2pio = '/mnt/nas2/data/systematicReview/UMLS/english_subset/umls_preprocessed/tui_pio.tsv'

#cui2tuiMapper(indir, outdir, f_tui2pio)