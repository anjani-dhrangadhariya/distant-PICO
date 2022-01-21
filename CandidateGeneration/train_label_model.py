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

def lf_levels(umls_d:dict, pattern:str, picos:str):

    umls_level = dict()

    for key, value in umls_d.items():   # iter on both keys and values
        search_pattern = pattern + picos
        if key.startswith(search_pattern):
            k = str(key).split('_')[-1]
            umls_level[ k ] = value

    return umls_level

# Level 1: UMLS
umls_p = lf_levels(lfs, 'UMLS_fuzzy_', 'p')
umls_i = lf_levels(lfs, 'UMLS_fuzzy_', 'i')
umls_o = lf_levels(lfs, 'UMLS_fuzzy_', 'o')

# Level 2: non UMLS
nonumls_p = lf_levels(lfs, 'nonUMLS_direct_', 'P')
nonumls_i = lf_levels(lfs, 'nonUMLS_direct_', 'I')
nonumls_o = lf_levels(lfs, 'nonUMLS_direct_', 'O')

# Level 3: DS
ds_p = lf_levels(lfs, 'DS_fuzzy_', 'P')
ds_i = lf_levels(lfs, 'DS_fuzzy_', 'I')
ds_o = lf_levels(lfs, 'DS_fuzzy_', 'O')

# Level 4: dictionary, rules, heuristics
heur_p = lf_levels(lfs, 'heuristics_direct_', 'P')
heur_i = lf_levels(lfs, 'heuristics_direct_', 'I')
heur_o = lf_levels(lfs, 'heuristics_direct_', 'O')

heur_p = lf_levels(lfs, 'dictionary_direct_', 'P')
heur_i = lf_levels(lfs, 'dictionary_direct_', 'I')
heur_o = lf_levels(lfs, 'dictionary_direct_', 'O')


# Rank and partition UMLS SAB's
ranked_umls_p, partitioned_umls_p = rankSAB( umls_p, 'p' )
ranked_umls_i, partitioned_umls_i = rankSAB( umls_i, 'i' )
ranked_umls_o, partitioned_umls_o = rankSAB( umls_o, 'o' )

# for i, partition in enumerate(partitioned_umls_p):
#     print('Labeling function number: ', len(partition) )
#     print(partition)

# for i, partition in enumerate(partitioned_umls_i):
#     print('Labeling function number: ', len(partition) )

# for i, partition in enumerate(partitioned_umls_o):
#     print('Labeling function number: ', len(partition) )

# Keep the rest stagnant and change UMLS for training PIO * (partition functions) label models
# build the full pipeline before executing it....