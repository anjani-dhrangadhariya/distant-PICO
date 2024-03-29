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

    rawcand_file = f'/mnt/nas2/results/Results/systematicReview/distant_pico/predictions/LabelModels/i/v4/UMLS/42/stpartition_9_epoch_100_bestmodel_train.tsv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-entity', type = str, default = 'outcome') # participant, intervention, outcome, study type
    parser.add_argument('-rawcand_file', type = Path, default = rawcand_file )
    parser.add_argument('-ebm_nlp', type = Path, default = '/mnt/nas2/data/systematicReview/PICO_datasets/EBM_parsed/train_ebm.json')
    parser.add_argument('-ebm_gold', type = Path, default = '/mnt/nas2/data/systematicReview/PICO_datasets/EBM_parsed/test_ebm.json')
    parser.add_argument('-ebm_gold_corr', type = Path, default = '/mnt/nas2/data/systematicReview/PICO_datasets/EBM_parsed/test_ebm_anjani.json')

    # file saving arguments from the rawcand_file
    ent = str(rawcand_file).split('/')[9]
    version = str(rawcand_file).split('/')[10]
    exp_level = str(rawcand_file).split('/')[11]
    seed = str(rawcand_file).split('/')[12]
    #seed = 42

    parser.add_argument('-ent', type = str, default = ent)
    parser.add_argument('-version', type = str, default = version)
    parser.add_argument('-exp_level', type = str, default = exp_level)
    parser.add_argument('-seed', type = str, default = seed)

    # parser.add_argument('-ebm_nlp', type = Path, default = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/groundtruth/ebm_nlp/p/sentences.txt')
    # parser.add_argument('-ebm_gold', type = Path, default = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/groundtruth/ebm_gold/p/sentences.txt')
    # parser.add_argument('-hilfiker', type = Path, default = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/groundtruth/hilfiker/p/sentences.txt')

    parser.add_argument('-supervision', type = str, default = 'ws') # label_type = {fs, ws, hs, ...} 
    parser.add_argument('-label_type', type = str, default = 'seq_lab') # label_type = {seq_lab, BIO, BIOES, ...} 
    parser.add_argument('-text_level', type = str, default = 'sentence') # text_level = {sentence, document} 
    parser.add_argument('-train_data', type = str, default = 'ebm-pico') # train_data = {distant-cto, combined, ebm-pico}

    parser.add_argument('-train_from_scratch', type = str, default=True)
    parser.add_argument('-plugin_model', type = Path, default = '/mnt/nas2/results/Results/systematicReview/distant_pico/models/bertcrf/0_3.pth')

    parser.add_argument('-max_len', type = int, default=510)
    parser.add_argument('-num_labels', type = int, default = 2) # 2 for binary (O-span vs. P/I/O) classification, 5 for multiclass (PICO) classification

    parser.add_argument('-embed_type', type = str, default = 'contextual') # embed_type = {contextual, semantic} 
    parser.add_argument('-embed', type = str, default = 'BioLinkBERT') # embed = {scibert, bert, biobert, pubmedbert, BioLinkBERT ...} 
    parser.add_argument('-model', type = str, default = 'transformercrf') # model = {transformercrf, scibertposcrf, scibertposattencrf, ...} 
    parser.add_argument('-bidrec', type = str, default=True)

    parser.add_argument('-max_eps', type = int, default= 15)
    parser.add_argument('-loss', type = str, default = 'general')
    parser.add_argument('-freeze_bert', action='store_false') # store_false = won't freeze BERT
    parser.add_argument('-lr', type = float, default= 5e-4)
    parser.add_argument('-eps', type = float, default= 1e-8)
    parser.add_argument('-lr_warmup', type = float, default= 0.1) 


    parser.add_argument('-parallel', type = str, default = 'false') # false = won't use data parallel
    parser.add_argument('-gpu', type = int, default = device)

    parser.add_argument('-print_every', type = int, default= 100)
    parser.add_argument('-mode', type = str, default= "train")

    args = parser.parse_args()


    return args
