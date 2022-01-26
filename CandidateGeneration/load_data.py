import pandas as pd
import json
import argparse

from CandGenUtilities.experiment_arguments import *
from CandGenUtilities.labeler_utilities import *
from CandGenUtilities.source_target_mapping import *
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
    Loads EBM-NLP training set with PICO annotations, splits it into training and validation sets, and returns validation set

Args:
    train_dir (str): String containing path to the EBM-NLP directory

Returns:
    pandas data frame: a pandas dataframe for EBM-NLP training and validation sets
'''
def loadEBMPICO(train_dir):

    pmid = []
    text = []
    tokens = []
    pos = []
    p = []
    i = []
    o = []

    with open(f'{train_dir}/result.json', 'r') as rf:
        data = json.load(rf)
        for k,v in data.items():
            pmid.append( k )
            #text.extend( [x.strip() for x in v['tokens'] ] )
            tokens.append( [x.strip() for x in v['tokens'] ] )
            pos.append( v['pos'] )
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
    

    df = pd.DataFrame( {'pmid': pmid, 'tokens': tokens, 'pos': pos, 'p': p, 'i': i, 'o': o } )
    #train, validation = train_test_split(df, test_size=0.20)

    return df