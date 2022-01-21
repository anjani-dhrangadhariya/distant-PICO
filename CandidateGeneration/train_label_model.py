import enum
import os
import glob
from pathlib import Path
import pandas as pd

from Ontologies.ontoUtils import rankSAB


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

def lf_levels(umls_d, picos):

    umls_level = dict()

    for key, value in umls_d.items():   # iter on both keys and values
        search_pattern = 'UMLS_fuzzy_' + picos
        if key.startswith(search_pattern):
            k = str(key).split('_')[-1]
            umls_level[ k ] = value

    return umls_level

umls_p = lf_levels(lfs, 'p')
umls_i = lf_levels(lfs, 'i')
umls_o = lf_levels(lfs, 'o')

# Combine UMLS candidate labels into partitions
# Rank the ontology based on coverage on the validation set
# Combine the ontologies into labeling functions
ranked_umls_p, partitioned_umls_p = rankSAB( umls_p, 'p' )
ranked_umls_i, partitioned_umls_i = rankSAB( umls_i, 'i' )
ranked_umls_o, partitioned_umls_o = rankSAB( umls_o, 'o' )

for i, partition in enumerate(partitioned_umls_p):
    print('Labeling function number: ', len(partition) )
    print(partition)

# for i, partition in enumerate(partitioned_umls_i):
#     print('Labeling function number: ', len(partition) )

# for i, partition in enumerate(partitioned_umls_o):
#     print('Labeling function number: ', len(partition) )

# Keep the rest stagnant and change UMLS for training PIO * (partition functions) label models

# build the full pipeline before executing it....