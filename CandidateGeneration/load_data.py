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
def loadEBMPICO(train_dir, outdir, write_to_file):

    outdir = str( outdir )

    pmid = []
    text = []
    tokens = []
    pos = []
    char_offsets = []
    p = []
    i = []
    o = []

    filename = ''
    if 'train' in outdir:
        filename = 'train_ebm.json'
        write_file = 'train_ebm_labels_tui_pio3.tsv'
    if 'test' in outdir and 'ebm' in outdir:
        filename = 'test_ebm.json'
        write_file = 'test_ebm_labels_tui_pio3.tsv'
    if 'test' in outdir and 'physio' in outdir:
        filename = 'test_physio.json'
        write_file = 'test_physio_labels_tui_pio3.tsv'

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

            if 'interventions' in v:
                vi = v['interventions']
                i.append( vi )
            else:
                i.append( [ '0' ] * len( v['tokens'] ) )

            if 'outcomes' in v:
                vo = v['outcomes']
                o.append( vo  )
            else:
                o.append( [ '0' ] * len( v['tokens'] ) )
    

    df_data = pd.DataFrame( {'pmid': pmid, 'tokens': tokens, 'pos': pos, 'offsets': char_offsets, 'p': p, 'i': i, 'o': o } )
    
    text = get_text(df_data['tokens'], df_data['offsets'])
    df_data['text'] = text

    df_data_token_flatten = [item for sublist in list(df_data['tokens']) for item in sublist]
    df_data_pos_flatten = [item for sublist in list(df_data['pos']) for item in sublist]
    df_data_offset_flatten = [item for sublist in list(df_data['offsets']) for item in sublist]

    df_data_p_labels_flatten = [item for sublist in list(df_data['p']) for item in sublist]
    df_data_p_labels_flatten = list(map(int, df_data_p_labels_flatten))

    df_data_i_labels_flatten = [item for sublist in list(df_data['i']) for item in sublist]
    df_data_i_labels_flatten = list(map(int, df_data_i_labels_flatten))

    df_data_o_labels_flatten = [item for sublist in list(df_data['o']) for item in sublist]
    df_data_o_labels_flatten = list(map(int, df_data_o_labels_flatten))


    write_df = pd.DataFrame(
    {
    'tokens': df_data_token_flatten,
    'pos': df_data_pos_flatten,
    'offsets': df_data_offset_flatten,
    'p': df_data_p_labels_flatten,
    'i': df_data_i_labels_flatten,
    'o': df_data_o_labels_flatten,
    })

    if write_to_file == True:
        df_data.to_csv(f'/mnt/nas2/results/Results/systematicReview/distant_pico/EBM_PICO_GT/v3/{write_file}', sep='\t')

    return df_data, write_df