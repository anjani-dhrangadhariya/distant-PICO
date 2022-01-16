import pandas as pd
import json
import argparse

from sklearn.model_selection import train_test_split

from CandGenUtilities.experiment_arguments import *
from CandGenUtilities.labeler_utilities import *
from CandGenUtilities.source_target_mapping import *
from LabelingFunctions.ontologyLF import *
from Ontologies.ontologyLoader import *
from sanity_checks import *

################################################################################
# Initialize and set seed
################################################################################
# Get the experiment arguments
args = getArguments()

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
def load_validation_set(train_dir):
    corpus = []
    corpus_labels = []
    corpus_pos = []

    with open(f'{train_dir}/{args.entity}/sentence_annotation2POS.txt', 'r') as rf:
        for eachStudy in rf:
            data = json.loads(eachStudy)
            
            for k,v in data.items():

                sentence_tokens = []
                sentence_labels = []
                sentence_pos = []

                for sent_id, sent in v.items():

                    sentence_tokens.extend( sent[0] )
                    sentence_labels.extend( sent[1] )
                    sentence_pos.extend( sent[2] )

                corpus.append( [x.strip() for x in sentence_tokens ] )
                corpus_labels.append( [x.strip() for x in sentence_labels ] )
                corpus_pos.append( [x.strip() for x in sentence_pos ] )
    
    df = pd.DataFrame( {'text': corpus, 'labels': corpus_labels, 'pos': corpus_pos} )

    train, validation = train_test_split(df, test_size=0.20)

    return train, validation