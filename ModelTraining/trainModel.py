import os
import random
import time
import traceback

import mlflow
import numpy as np
import pandas as pd

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlflow import log_artifacts, log_metric, log_param
from torch import LongTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaConfig, RobertaModel
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from transformers import AdamW 
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()

from read_candidates import fetchAndTransformCandidates
from Utilities.choosers import *
from Utilities.mlflow_logging import *
from train_eval_predict import *


def convertDf2Tensor(df, data_type):

    if data_type==np.float64:

        # return torch.from_numpy( np.array( list( [ [0.0, 0.0], [0.0, 0.0] ] ) , dtype=data_type ) ).clone().detach()
        return torch.from_numpy( np.array( list( df ), dtype=float ) ).clone().detach()

    else:
        return torch.from_numpy( np.array( list( df ), dtype=data_type ) ).clone().detach()

def loadModel(model, exp_args):

    if exp_args.parallel == 'true':

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids = [0, 1])
            print("Using", str(model.device_ids), " GPUs!")
            return model

    elif exp_args.parallel == 'false':
        model = nn.DataParallel(model, device_ids = [0])
        return model

if __name__ == "__main__":

    try:

        # for eachSeed in [ 0, 1, 42 ]:
        for eachSeed in [ 0 ]:

            with mlflow.start_run() as run:

                def seed_everything( seed ):
                    random.seed(seed)
                    os.environ['PYTHONHASHSEED'] = str(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.deterministic = True
                seed_everything(eachSeed)

                print('The random seed is set to: ', eachSeed)

                # This is executed after the seed is set because it is imperative to have reproducible data run after shuffle
                weak_candidates, ebm_train_df, ebm_gold_df, ebm_gold_corr_df, exp_args, current_tokenizer, current_modelembed = fetchAndTransformCandidates()

                # # shuffle the dataset and divide into training and validation sets
                # fulldf = fulldf.sample(frac=1).reset_index(drop=True)
                if exp_args.supervision == 'ws': 
                    annotations_train_df, annotations_dev_df = train_test_split(weak_candidates, test_size=0.2) 
                elif exp_args.supervision == 'fs': 
                    annotations_train_df, annotations_dev_df = train_test_split(ebm_train_df, test_size=0.2) 

                print('Size of training set: ', len(annotations_train_df.index))
                print('Size of development set: ', len(annotations_dev_df.index))


                # # Log the parameters
                # logParams(exp_args, eachSeed)

                # Convert all inputs, labels, and attentions into torch tensors, the required datatype: torch.int64
                train_input_ids = convertDf2Tensor(annotations_train_df['embeddings'], np.int64)
                if exp_args.supervision == 'ws': 
                    train_input_labels = convertDf2Tensor(annotations_train_df['label_pads'], np.float64)
                elif exp_args.supervision == 'fs': 
                    train_input_labels = convertDf2Tensor(annotations_train_df['label_pads'], np.int64)
                train_attn_masks = convertDf2Tensor(annotations_train_df['attn_masks'], np.int64)
                train_pos_tags = convertDf2Tensor(annotations_train_df['inputpos'], np.int64)
                # train_pos_tags = torch.nn.functional.one_hot( torch.from_numpy( np.array( list(annotations_train_df['inputpos']), dtype=np.int64) ).clone().detach() )
                # assert train_input_ids.dtype == train_input_labels.dtype == train_attn_masks.dtype == train_pos_tags.dtype

                # Test set (EBM-NLP training data used as test set)
                dev_input_ids = convertDf2Tensor( annotations_dev_df['embeddings'], np.int64)
                if exp_args.supervision == 'ws': 
                    dev_input_labels = convertDf2Tensor( annotations_dev_df['label_pads'], np.float64)
                elif exp_args.supervision == 'fs': 
                    dev_input_labels = convertDf2Tensor( annotations_dev_df['label_pads'], np.int64)
                dev_attn_masks = convertDf2Tensor( annotations_dev_df['attn_masks'], np.int64)
                dev_pos_tags = convertDf2Tensor( annotations_dev_df['attn_masks'], np.int64)
                # test_pos_tags = torch.nn.functional.one_hot( torch.from_numpy( np.array( list(annotations_dev_df['inputpos']))))
                # assert dev_input_ids.dtype == dev_input_labels.dtype == dev_attn_masks.dtype == dev_pos_tags.dtype

                # Test set 1 (EBM-NLP test gold data test set)
                test1_input_ids = convertDf2Tensor( ebm_gold_df['embeddings'], np.int64)
                test1_input_labels = convertDf2Tensor( ebm_gold_df['label_pads'], np.int64)
                test1_attn_masks = convertDf2Tensor( ebm_gold_df['attn_masks'], np.int64)
                test1_pos_tags = convertDf2Tensor( ebm_gold_df['attn_masks'], np.int64)

                # Test set 2 (EBM-NLP test gold CORRECTED data test set)
                test2_input_ids = convertDf2Tensor( ebm_gold_corr_df['embeddings'], np.int64)
                test2_input_labels = convertDf2Tensor( ebm_gold_corr_df['label_pads'], np.int64)
                test2_attn_masks = convertDf2Tensor( ebm_gold_corr_df['attn_masks'], np.int64)
                test2_pos_tags = convertDf2Tensor( ebm_gold_corr_df['attn_masks'], np.int64)

                # # ----------------------------------------------------------------------------------------
                # # Create dataloaders from the tensors
                # # ----------------------------------------------------------------------------------------
                # # Create the DataLoader for our training set.
                train_data = TensorDataset(train_input_ids, train_input_labels, train_attn_masks, train_pos_tags)
                train_sampler = RandomSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=None, batch_size=10, shuffle=False)

                # # Create the DataLoader for our test set. (This will be used as validation set!)
                dev_data = TensorDataset(dev_input_ids, dev_input_labels, dev_attn_masks, dev_pos_tags)
                dev_sampler = RandomSampler(dev_data)
                dev_dataloader = DataLoader(dev_data, sampler=None, batch_size=10, shuffle=False)

                # Create the DataLoader for our test set 1.
                test1_data = TensorDataset(test1_input_ids, test1_input_labels, test1_attn_masks, test1_pos_tags)
                test1_sampler = RandomSampler(test1_data)
                test1_dataloader = DataLoader(test1_data, sampler=None, batch_size=6, shuffle=False)

                # Create the DataLoader for our test set 2.
                test2_data = TensorDataset(test2_input_ids, test2_input_labels, test2_attn_masks, test2_pos_tags)
                test2_sampler = RandomSampler(test2_data)
                test2_dataloader = DataLoader(test2_data, sampler=None, batch_size=6, shuffle=False)

                # ##################################################################################
                # #Instantiating the BERT model
                # ##################################################################################
                print("Building model...")
                createOSL = time.time()
                model = choose_model(exp_args.embed, current_tokenizer, current_modelembed, exp_args.model, exp_args)

                # ##################################################################################
                # # Tell pytorch to run data on this model on the GPU and parallelize it
                # ##################################################################################

                if exp_args.train_from_scratch == True:
                    model = loadModel(model, exp_args)
                else:
                    checkpoint = torch.load(exp_args.plugin_model, map_location='cuda:0')
                    model.load_state_dict( checkpoint )
                    model = loadModel(model, exp_args)
                print('The devices used: ', str(model.device_ids) )

                # ##################################################################################
                # # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
                optimizer = AdamW(model.parameters(),
                                lr = exp_args.lr, # args.learning_rate - default is 5e-5 (for BERT-base)
                                eps = exp_args.eps, # args.adam_epsilon  - default is 1e-8.
                                )

                # Total number of training steps is number of batches * number of epochs.
                total_steps = len(train_dataloader) * exp_args.max_eps
                print('Total steps per epoch: ', total_steps)

                # Create the learning rate scheduler.
                scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                            num_warmup_steps=total_steps*exp_args.lr_warmup,
                                                            num_training_steps = total_steps)

                # print("Created the optimizer, scheduler and loss function objects in {} seconds".format(time.time() - st))
                print("--- Took %s seconds to create the model, optimizer, scheduler and loss function objects ---" % (time.time() - createOSL))

                # print('##################################################################################')
                # print('Begin training...')
                # print('##################################################################################')
                train_start = time.time()
                saved_models = train(model, optimizer, scheduler, train_dataloader, dev_dataloader, exp_args, run, eachSeed)
                print("--- Took %s seconds to train and evaluate the model ---" % (time.time() - train_start))

                # # Use the saved models to evaluate on the test set
                # # print( saved_models )

                print('##################################################################################')
                print('Begin test...')
                print('##################################################################################')
                checkpoint = torch.load(saved_models[-1], map_location='cuda:0')
                # checkpoint = torch.load('/mnt/nas2/results/Results/systematicReview/distant_pico/FS/participant/transformercrf/0_7.pth', map_location='cuda:0')
                model.load_state_dict( checkpoint )
                model = torch.nn.DataParallel(model, device_ids=[0])

                print('Applying the best model on test set (EBM-NLP)...')
                test1_cr, all_pred_flat, all_GT_flat, cm1    = evaluate(model, optimizer, scheduler, test1_dataloader, exp_args, mode='test')
                print(test1_cr)

                print('Applying the best model on test set (EBM-NLP corrected)...')
                test2_cr, all_pred_flat, all_GT_flat, cm2 = evaluate(model, optimizer, scheduler, test2_dataloader, exp_args, mode='test')
                print(test2_cr)

    except Exception as ex:

        template = "An exception of type {0} occurred. Arguments:{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print( message )

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(traceback.format_exc())

        logging.info(message)
        string2log = str(exc_type) + ' : ' + str(fname) + ' : ' + str(exc_tb.tb_lineno)
        logging.info(string2log)
        logging.info(traceback.format_exc())