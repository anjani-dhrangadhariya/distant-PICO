# Transformers 
from transformers import (AutoTokenizer, BertConfig, BertModel, BertTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer, RobertaConfig,
                          RobertaModel, RobertaTokenizer)


##################################################################################
# Load the chosen tokenizer
##################################################################################
def choose_tokenizer_type(pretrained_model):
    
    print('Loading tokenizer...')
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
