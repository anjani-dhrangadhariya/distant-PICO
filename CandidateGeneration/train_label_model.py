import os
import glob
from pathlib import Path
import pandas as pd


indir = '/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation'
pathlist = Path(indir).glob('**/*.tsv')

tokens = []

lfs = dict()

for file in pathlist:
    
    k = str( file ).split('candidate_generation/')[-1].replace('.tsv', '').replace('/', '_')
    data = pd.read_csv(file, sep='\t', header=0)
    if len(tokens) == 0:
        tokens.extend( list(data.tokens) )
    
    lfs[str(k)] = list(data.columns[-1])


print( 'Total number of tokens in validation set: ', len(tokens) )
print( 'Total number of LFs in the dictionary', len(lfs) )


# Combine UMLS candidate labels into partitions

# Keep the rest stagnant and change UMLS for training PIO * (partition functions) label models

# build the full pipeline before executing it....