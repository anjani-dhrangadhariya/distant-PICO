import os
import glob
from pathlib import Path
import pandas as pd


indir = '/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation'
pathlist = Path(indir).glob('**/*.tsv')

tokens = []

for file in pathlist:
    
    k = str( file ).split('candidate_generation/')[-1].replace('.tsv', '')
    print(k)

    data = pd.read_csv(file, sep='\t', header=0)
    if len(tokens) == 0:
        tokens.extend( list(data.tokens) )


print( 'Total number of tokens in validation set: ', len(tokens) )




# XXX: doing - Read all the labeled candidates into a dictionary

# Combine UMLS candidate labels into partitions

# Keep the rest stagnant and change UMLS for training PIO * (partition functions) label models

# build the full pipeline before executing it....