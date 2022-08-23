from re import S
import pandas as pd
import json
import argparse

from CandGenUtilities.experiment_arguments import *
from CandGenUtilities.labeler_utilities import *
from CandGenUtilities.source_target_mapping import *
from LabelingFunctions.LFutils import get_text
from LabelingFunctions.ontologyLF import *
from Ontologies.ontologyLoader import *

################################################################################
# Initialize and set seed
################################################################################
seed = 0
seed_everything(seed)
print('The random seed is set to: ', seed)

'''
Description:
    Loads EBM-NLP training set with PICO annotations and returns 

Args:
    train_dir (str): String containing path to the EBM-NLP directory
    write_to_file (bool): switch to write training set to file

Returns:
    Formatted training text, tokens and token labels (str, df, df, df, ,df, df)
'''
def loadEBMPICO(train_dir, outdir, candgen_version, write_to_file):

    outdir = str( outdir )

    pmid = []
    text = []
    tokens = []
    pos = []
    char_offsets = []
    p = []
    p_f = []
    i = []
    i_f = []
    o = []
    o_f = []
    s = []
    s_f = []

    filename = ''
    if 'train' in outdir and 'anjani' not in outdir:
        filename = 'train_ebm.json'
        write_file = 'train_ebm_labels_tui_pio3.tsv'
    if 'test' in outdir and 'ebm' in outdir and 'anjani' not in outdir:
        filename = 'test_ebm.json'
        write_file = 'test_ebm_labels_tui_pio3.tsv'
    if 'test' in outdir and 'physio' in outdir and 'anjani' not in outdir:
        filename = 'test_physio.json'
        write_file = 'test_physio_labels_tui_pio3.tsv'
    if 'test' in outdir and 'ebm' in outdir and 'anjani' in outdir:
        filename = 'test_ebm_anjani.json'
        write_file = 'test_ebm_correctedlabels_tui_pio3.tsv'

    with open(f'{train_dir}/{filename}', 'r') as rf:
        data = json.load(rf)
        for k,v in data.items():
            pmid.append( k )
            tokens.append( [x.strip() for x in v['tokens'] ] )
            pos.append( v['pos'] )
            char_offsets.append( v['abs_char_offsets'] )
            
            if 'participants' in v:
                vp = v['participants']
                p.append( vp )
            else:
                p.append( [ '0' ] * len( v['tokens'] ) )

            if 'participants_fine' in v:
                vp_f = v['participants_fine']
                # replace O (alphabet) with 0
                if 'O' in vp_f:
                    vp_f = ['0' if x == 'O' else x for x in vp_f ]
                p_f.append( vp_f )
            else:
                p_f.append( [ '0' ] * len( v['tokens'] ) )


            if 'interventions' in v:
                vi = v['interventions']
                i.append( vi )
            else:
                i.append( [ '0' ] * len( v['tokens'] ) )

            if 'interventions_fine' in v:
                vi_f = v['interventions_fine']
                if 'O' in vi_f:
                    vi_f = ['0' if x == 'O' else x for x in vi_f ]
                i_f.append( vi_f )
            else:
                i_f.append( [ '0' ] * len( v['tokens'] ) )


            if 'outcomes' in v:
                vo = v['outcomes']
                o.append( vo  )
            else:
                o.append( [ '0' ] * len( v['tokens'] ) )

            if 'outcomes_fine' in v:
                vo_f = v['outcomes_fine']
                if 'O' in vo_f:
                    vo_f = ['0' if x == 'O' else x for x in vo_f ]
                o_f.append( vo_f  )
            else:
                o_f.append( [ '0' ] * len( v['tokens'] ) )


            if 'studytype' in v:
                vs = v['studytype']
                s.append( vs  )
            else:
                s.append( [ '0' ] * len( v['tokens'] ) )

            if 'studytype_fine' in v:
                vs_f = v['studytype']
                if 'O' in vs_f:
                    vs_f = ['0' if x == 'O' else x for x in vs_f ]
                s_f.append( vs_f  )
            else:
                s_f.append( [ '0' ] * len( v['tokens'] ) )
    

    df_data = pd.DataFrame( {'pmid': pmid, 'tokens': tokens, 'pos': pos, 'offsets': char_offsets, 'p': p, 'i': i, 'o': o, 's': s, 'p_f': p_f, 'i_f': i_f, 'o_f': o_f, 's_f': s_f  } )
    
    text = get_text(df_data['tokens'], df_data['offsets'])
    df_data['text'] = text

    df_data_token_flatten = [item for sublist in list(df_data['tokens']) for item in sublist]
    df_data_pos_flatten = [item for sublist in list(df_data['pos']) for item in sublist]
    df_data_offset_flatten = [item for sublist in list(df_data['offsets']) for item in sublist]

    df_data_p_labels_flatten = [item for sublist in list(df_data['p']) for item in sublist]
    df_data_p_labels_flatten = list(map(int, df_data_p_labels_flatten))

    df_data_pf_labels_flatten = [item for sublist in list(df_data['p_f']) for item in sublist]
    df_data_pf_labels_flatten = list(map(int, df_data_pf_labels_flatten))

    #############################################

    df_data_i_labels_flatten = [item for sublist in list(df_data['i']) for item in sublist]
    df_data_i_labels_flatten = list(map(int, df_data_i_labels_flatten))

    df_data_if_labels_flatten = [item for sublist in list(df_data['i_f']) for item in sublist]
    df_data_if_labels_flatten = list(map(int, df_data_if_labels_flatten))

    #############################################

    df_data_o_labels_flatten = [item for sublist in list(df_data['o']) for item in sublist]
    df_data_o_labels_flatten = list(map(int, df_data_o_labels_flatten))

    df_data_of_labels_flatten = [item for sublist in list(df_data['o_f']) for item in sublist]
    df_data_of_labels_flatten = list(map(int, df_data_of_labels_flatten))

    #############################################

    df_data_s_labels_flatten = [item for sublist in list(df_data['s']) for item in sublist]
    df_data_s_labels_flatten = list(map(int, df_data_o_labels_flatten))

    df_data_sf_labels_flatten = [item for sublist in list(df_data['s_f']) for item in sublist]
    df_data_sf_labels_flatten = list(map(int, df_data_sf_labels_flatten))

    # print( len(df_data_sf_labels_flatten) )


    write_df = pd.DataFrame(
    {
    'tokens': df_data_token_flatten,
    'pos': df_data_pos_flatten,
    'offsets': df_data_offset_flatten,
    'p': df_data_p_labels_flatten,
    'p_f': df_data_pf_labels_flatten,
    'i': df_data_i_labels_flatten,
    'i_f': df_data_if_labels_flatten,
    'o': df_data_o_labels_flatten,
    'o_f': df_data_of_labels_flatten,
    's': df_data_s_labels_flatten,
    's_f': df_data_sf_labels_flatten,
    })

    if write_to_file == True:
        df_data.to_csv(f'/mnt/nas2/results/Results/systematicReview/distant_pico/EBM_PICO_GT/{candgen_version}/gt/{write_file}', sep='\t')

    return df_data, write_df