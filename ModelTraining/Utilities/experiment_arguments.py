import argparse
from pathlib import Path

# pyTorch essentials
import torch

##################################################################################
# set up the GPU
##################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print('Number of GPUs identified: ', n_gpu)
print('You are using ', torch.cuda.get_device_name(0), ' : ', device , ' device')

def getArguments():

    # List of arguments to set up experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_file', type = Path, default = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/PICOS_data_preprocessed/merged_1_0.txt')
    parser.add_argument('-embed', type = str, default = 'bert') # embed = {scibert, bert, biobert, ...} 
    parser.add_argument('-embed_type', type = str, default = 'contextual') # embed_type = {contextual, semantic} 
    parser.add_argument('-model', type = str, default = 'bertcrf') # model = {scibertposcrf, scibertposattencrf, ...} 
    parser.add_argument('-label_type', type = str, default = 'seq_lab') # label_type = {seq_lab, BIO, BIOES, ...} 
    parser.add_argument('-text_level', type = str, default = 'document') # text_level = {sentence, document} 
    parser.add_argument('-train_data', type = str, default = 'distant-cto') # train_data = {distant-cto, combined, ebm-pico} 
    parser.add_argument('-parallel', type = str, default = 'false') # false = won't use data parallel
    parser.add_argument('-gpu', type = int, default = device)
    parser.add_argument('-freeze_bert', action='store_false') # store_false = won't freeze BERT
    parser.add_argument('-print_every', type = int, default= 200)
    parser.add_argument('-max_eps', type = int, default= 10)
    parser.add_argument('-lr', type = float, default= 5e-4)
    parser.add_argument('-eps', type = float, default= 1e-8)
    parser.add_argument('-loss', type = str, default = 'general')
    parser.add_argument('-bidrec', type = str, default=True)
    parser.add_argument('-max_len', type = int, default=100)

    args = parser.parse_args()


    return args
