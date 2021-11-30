# Transformers 
from Models.TRANSFORMER_CRF import TRANSFORMERCRF
from transformers import (AutoModel, AutoModelWithLMHead,
                          AutoTokenizer, AutoConfig, AutoModelForTokenClassification)


##################################################################################
# Load the chosen tokenizer
##################################################################################
def choose_tokenizer_type(pretrained_model):
    
    if pretrained_model == 'bert':
        # tokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokenizer_ = AutoTokenizer.from_pretrained('bert-base-uncased')
        model_ = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=False)

    elif 'gpt2' in pretrained_model:
        # tokenizer_ = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True, unk_token="<|endoftext|>")
        tokenizer_ = AutoTokenizer.from_pretrained("gpt2", do_lower_case=True, unk_token="<|endoftext|>")
        model_ = AutoModel.from_pretrained("gpt2")

    elif 'biobert' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        model_ = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

    elif 'scibert' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        model_ = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")

    elif 'roberta' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("roberta-base")
        model_ = AutoModel.from_pretrained("roberta-base")

    elif 'pubmedbert' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        model_ = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", output_hidden_states=True, output_attentions=False)

    return tokenizer_ , model_

##################################################################################
# The function loads the chosen model
##################################################################################
def choose_model(vector_type, tokenizer, modelembed, chosen_model, args):

    if chosen_model == 'transformercrf':
        model = TRANSFORMERCRF(args.freeze_bert, tokenizer, modelembed, args)
    # elif chosen_model == 'bertbilstmcrf':
    #     model = BERTBiLSTMCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif chosen_model == 'bertlinear':
    #     model = BERTLinear(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif chosen_model == 'scibertcrf':
    #     model = SCIBERTCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif chosen_model == 'scibertposcrf':
    #     model = SCIBERTPOSCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif chosen_model == 'scibertposlinear':
    #     model = SCIBERTPOSLinear(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif chosen_model == 'scibertlinear':
    #     model = SCIBERTLinear(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif chosen_model == 'scibertposattencrf':
    #     model = SCIBERTPOSAttenCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif chosen_model == 'scibertposattenlinear':
    #     model = SCIBERTPOSAttenLinear(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif chosen_model == 'scibertposattenact':
    #     model = SCIBERTPOSAttenActLin(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif chosen_model == 'semanticcrf':
    #     model = SemanticCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    else:
        print('Please enter correct model name...')

    return model
