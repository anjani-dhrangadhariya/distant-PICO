import enum
import glob
import os
from hashlib import new
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from flyingsquid.label_model import LabelModel as LMsquid
from sklearn.model_selection import train_test_split
from snorkel.labeling.model import LabelModel as LMsnorkel
from snorkel.labeling.model import MajorityLabelVoter

import LMutils

file = '/mnt/nas2/results/Results/systematicReview/distant_pico/EBM_PICO_GT/validation_labels.tsv'
df_data = pd.read_csv(file, sep='\t', header=0)
train, validation = train_test_split(df_data, test_size=0.20, shuffle = False, stratify = None)

# Read Candidate labels from multiple LFs
indir = '/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation'
pathlist = Path(indir).glob('**/*.tsv')

tokens = []

lfs = dict()

for file in pathlist:
    
    k = str( file ).split('candidate_generation/')[-1].replace('.tsv', '').replace('/', '_')
    data = pd.read_csv(file, sep='\t', header=0)
    if len(tokens) == 0:
        tokens.extend( list(data.tokens) )
    
    sab = data.columns[-1]
    lfs[str(k)] = list( data[sab] )


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
nonumls_p = lf_levels(lfs, 'nonUMLS_fuzzy_', 'P')
nonumls_i = lf_levels(lfs, 'nonUMLS_fuzzy_', 'I')
nonumls_o = lf_levels(lfs, 'nonUMLS_fuzzy_', 'O')

# Level 3: DS
ds_p = lf_levels(lfs, 'DS_fuzzy_', 'P')
ds_i = lf_levels(lfs, 'DS_fuzzy_', 'I')
ds_o = lf_levels(lfs, 'DS_fuzzy_', 'O')

# Level 4: dictionary, rules, heuristics
heur_p = lf_levels(lfs, 'heuristics_direct_', 'P')
heur_i = lf_levels(lfs, 'heuristics_direct_', 'I')
heur_o = lf_levels(lfs, 'heuristics_direct_', 'O')

dict_p = lf_levels(lfs, 'dictionary_direct_', 'P')
dict_i = lf_levels(lfs, 'dictionary_direct_', 'I')
dict_o = lf_levels(lfs, 'dictionary_direct_', 'O')

# Rank and partition UMLS SAB's
ranked_umls_p, partitioned_umls_p = LMutils.rankSAB( umls_p, 'p' )
ranked_umls_i, partitioned_umls_i = LMutils.rankSAB( umls_i, 'i' )
ranked_umls_o, partitioned_umls_o = LMutils.rankSAB( umls_o, 'o' )

def getLFs(partition:list, umls_d:dict, seed_len:int):

    all_lfs_combined = []
    
    for lf in partition: # for each lf in a partition
        
        combine_here = [-1] *seed_len

        for sab in lf:
            new_a = umls_d[sab]
            old_a = combine_here
            temp_a = []
            for o_a, n_a in zip(old_a, new_a):
                replace_a = max( o_a, n_a )
                temp_a.append( replace_a )

            combine_here = temp_a

        all_lfs_combined.append( combine_here )

    return all_lfs_combined


'''#########################################################################
# Choosing the number of LF's from UMLS all
#########################################################################'''
for i, partition in enumerate(partitioned_umls_p):

    combined_lf = getLFs(partition, umls_p, len(Y_p))
    assert len(partition) == len(combined_lf)
 
    combined_lf.extend( list(nonumls_p.values()) ) # Combine with level 2
    combined_lf.extend( list(ds_p.values()) ) # Combine with level 3
    combined_lf.extend( list(heur_p.values()) ) # Combine with level 4
    combined_lf.extend( list(dict_p.values()) ) # combine with level 4

    print( len(combined_lf[0]) )


    # train model
    # L_train = np.array( combined_lf )
    # print(L_train.shape)
    # n = L_train.shape[1]
    # label_model = LMsquid(n)
    # label_model.fit(L_train)
    # preds = label_model.predict(L_train)

    # print( preds.shape )

    # pickle model
    # validate model

    if i== 0:
        break



'''#########################################################################
# Level 1 only
#########################################################################'''


'''#########################################################################
# Level 1 + Level 2
#########################################################################'''


'''#########################################################################
# Level 1 + Level 2 + Level 3
#########################################################################'''


'''#########################################################################
# Level 1 + Level 2 + Level 3 + Level 4
#########################################################################'''






# for i, partition in enumerate(partitioned_umls_i):
#     print('Labeling function number: ', len(partition) )

# for i, partition in enumerate(partitioned_umls_o):
#     print('Labeling function number: ', len(partition) )

# Keep the rest stagnant and change UMLS for training PIO * (partition functions) label models
# build the full pipeline before executing it....


#########################################################################################
# Combine LF's into a single LF
#########################################################################################
# L_p = [umls_p_labels, p_DO_labels, p_DO_syn_labels, p_ctd_labels, p_ctd_syn_labels, p_DS_labels, gender_labels, p_abb_labels, samplesize_labels, agerange_labels, agemax_labels, pa_regex_heur_labels, umls_p_fz_labels, p_DO_fz_labels, p_DO_syn_fz_labels, p_ctd_fz_labels, p_ctd_syn_fz_labels, p_DS_fz_labels ]
# L_i = [umls_i_labels, i_ctd_labels, i_ctd_syn_labels, i_chebi_labels, i_chebi_syn_labels, i_ds_labels, i_syn_ds_labels, comparator_labels, i_posreg_labels, umls_i_fz_labels, i_ctd_fz_labels, i_ctd_syn_fz_labels, i_chebi_fz_labels, i_chebi_syn_fz_labels, i_ds_fz_labels, i_syn_ds_fz_labels ]
# L_o = [umls_o_labels, o_oae_labels, o_oae_syn_labels, o_ds_labels, umls_o_fz_labels, o_oae_fz_labels, o_oae_syn_fz_labels, o_ds_fz_labels ]

# L_p = scipy.sparse.csr_matrix( L_p )
# participant_LF_summary = lf_summary(L_p, Y=validation_p_labels_flatten)
# print( participant_LF_summary )

# start_time = time.time()
# print('Training Label Model...')
# L = np.array(L_i)
# label_model.fit(L, seed=seed, n_epochs=100)
# print("--- %s seconds ---" % (time.time() - start_time))

# Y_hat = label_model.predict_proba(L)

# print( type(Y_hat) )
