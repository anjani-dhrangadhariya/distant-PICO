import argparse
from pathlib import Path

# pyTorch essentials
import torch

def getSources(annotations, lab):

    sources = []
    for k,v in annotations.items():

        more_than_source = len( v )
        for k_i, v_i in v.items():
            if 'source' in k_i and more_than_source > 1:
                sources.append( v_i )

    return sources, [lab] * len(sources)

def getArguments():

    # List of arguments to set up experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_example', type = Path, default = '/mnt/nas2/results/Results/systematicReview/distant_pico/models/bertcrf/0_3.pth')
    parser.add_argument('-labeler', type = str, default = 'contextual') # embed_type = {contextual, semantic} 
    parser.add_argument('-int_example', type = int, default= 10)
    parser.add_argument('-float_example', type = float, default= 1e-8)

    args = parser.parse_args()

    return args