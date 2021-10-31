# Transformers 
from transformers import (AutoTokenizer, BertConfig, BertModel, BertTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer, RobertaConfig,
                          RobertaModel, RobertaTokenizer)

from Models.BERT_CRF import BERTCRF


##################################################################################
# Load the chosen tokenizer
##################################################################################
def choose_tokenizer_type(pretrained_model):
    
    if 'bert' in pretrained_model and 'bio' not in pretrained_model and 'sci' not in pretrained_model:
        tokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    elif 'gpt2' in pretrained_model:
        tokenizer_ = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True, unk_token="<|endoftext|>")

    elif 'biobert' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

    elif 'scibert' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    elif 'roberta' in pretrained_model:
        tokenizer_ = RobertaTokenizer.from_pretrained("roberta-base")

    return tokenizer_

##################################################################################
# The function loads the chosen model
##################################################################################
def choose_model(vector_type, tokenizer, pretrained_model, args):

    if pretrained_model == 'bertcrf':
        model = BERTCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif pretrained_model == 'bertbilstmcrf':
    #     model = BERTBiLSTMCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif pretrained_model == 'bertlinear':
    #     model = BERTLinear(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif pretrained_model == 'scibertcrf':
    #     model = SCIBERTCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif pretrained_model == 'scibertposcrf':
    #     model = SCIBERTPOSCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif pretrained_model == 'scibertposlinear':
    #     model = SCIBERTPOSLinear(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif pretrained_model == 'scibertlinear':
    #     model = SCIBERTLinear(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif pretrained_model == 'scibertposattencrf':
    #     model = SCIBERTPOSAttenCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif pretrained_model == 'scibertposattenlinear':
    #     model = SCIBERTPOSAttenLinear(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif pretrained_model == 'scibertposattenact':
    #     model = SCIBERTPOSAttenActLin(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif pretrained_model == 'semanticcrf':
    #     model = SemanticCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    else:
        print('Please enter correct model name...')

    return model