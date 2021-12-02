#!/usr/bin/env python

def ParseOntDoc(a):
    '''The module maps CUIs to TUIs from the selected UMLS ontologies'''
    return a**a

print( ParseOntDoc.__doc__ )

import os
import errno
from collections import defaultdict
import msgpack
import csv

indir = '/mnt/nas2/data/systematicReview/UMLS/english_subset/2021AB/META'
outdir = '/mnt/nas2/data/systematicReview/UMLS/english_subset/umls_preprocessed'
tui2pio = '/mnt/nas2/data/systematicReview/UMLS/english_subset/umls_preprocessed/tui_pio.tsv'

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
    op.write('SAB\tTUI\tCUI\tTERM\n')
    for line in fp:
        row = line.strip().split('|')
        cui, sab, term = row[0], row[11], row[14]
        if term.strip() is None:
            continue
        for tui in cui_to_tui[cui]:
            op.write(f'{sab}\t{tui}\t{cui}\t{term}\t{tui_to_sty[tui]}\t{tui2pio_mapping[tui]}\n')


print('Data loaded')