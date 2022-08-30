# from CandidateGeneration.LabelingFunctions.heuristicLF import posPattern_i
# from LabelingFunctions.heuristicLF import posPattern_i
import collections
from multiprocessing.sharedctypes import Value
import os
import re
import sys

import Ontologies
import pandas as pd
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize, word_tokenize
from Ontologies.ontologyLoader import loadStopWords
from pytorch_pretrained_bert import BertTokenizer

# from CandidateGeneration.Ontologies.ontologyLoader import loadStopWords
from LabelingFunctions import heuristicLF, ontologyLF

pico2labelMap = { 'P' : 1, 'I' : 1, 'O' : 1, 'S': 1, '-P' : -1, '-I' : -1, '-O' : -1, '-S' : -1, '!P' : 0, '!I' : 0, '!O' : 0, '!S' : 0 }

# Load stopwords (generic negative label LFs)
sw = loadStopWords()

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

def get_text(words, offsets):

    S = []
    for w, o in zip( words, offsets ):
        s = ''
        for i, term in zip(o, w):
            if len(s) == i:
                s += str(term)
            elif len(s) < i:
                s += (' ' * (i - len(s))) + str(term)
            else:
                raise Exception('text offset error')
        
        S.append( s )

    return S

def match_term(term, dictionary, case_sensitive, lemmatize=True):
    """
    Parameters
    ----------
    term
    dictionary
    case_sensitive
    lemmatize   Including lemmas improves performance slightly
    Returns
    -------
    """
    if (not case_sensitive and term.lower() in dictionary) or term in dictionary:
        label_produced = None
        if (not case_sensitive and term.lower() in dictionary):
            label_produced = dictionary[ term.lower() ]
        if term in dictionary:
            label_produced = dictionary[ term ]
        return [True, label_produced]

    if (case_sensitive and lemmatize) and term.rstrip('s').lower() in dictionary:
        label_produced = dictionary[ term.rstrip('s').lower() ]
        return [True, label_produced]

    elif (not case_sensitive and lemmatize) and term.rstrip('s') in dictionary:
        label_produced = dictionary[ term.rstrip('s') ]
        return [True, label_produced]

    return [False, None]

def bert_tokenizer(text):
    tokens = tokenizer.tokenize(text)
    toks = []

    curr = ""
    for t in tokens:
        if t[0:2] == '##':
            curr = curr + t[2:]
        elif curr:
            toks.append(curr)
            curr = t
        elif not curr:
            curr = t
        else:
            toks.append(t)
            curr = ""
    if curr:
        toks.append(curr)

    return toks

def collection_to_dict(G_pos, picos, sign):

    label_dict = {}
    for k, v in dict(G_pos).items():
        for k_i, v_i in v.items():
            key = k + ' ' + k_i
            if '-' in sign:
                label_dict[ key ] = str(sign)+str(picos)
            else:
                label_dict[ key ] = str(picos)

    return label_dict

def build_word_graph(dictionary, picos, min_occur=25): # min_occur is a variable that can be tuned. Default = 25

    bigram_labels = {}

    G_pos = collections.defaultdict(collections.Counter)
    G_neg = collections.defaultdict(collections.Counter)

    for text, value in dictionary.items():
        tokens = bert_tokenizer(text)
        if len(tokens) == 1:
            continue
        if '-' not in value:
            for i in range(len(tokens)-1):
                G_pos[tokens[i]][tokens[i+1]] += 1
        else:
            for i in range(len(tokens)-1):
                G_neg[tokens[i]][tokens[i+1]] += 1

    if min_occur:
        for head in G_pos:
            rm = []
            for tail in G_pos[head]:
                if G_pos[head][tail] < min_occur:
                    rm.append(tail)
            for tail in rm:
                del G_pos[head][tail]

        for head in G_neg:
            rm = []
            for tail in G_neg[head]:
                if G_neg[head][tail] < min_occur:
                    rm.append(tail)
            for tail in rm:
                del G_neg[head][tail]

    # convert graph to dictionary with term:labels
    G_pos = collection_to_dict(G_pos, picos, '+')
    G_neg = collection_to_dict(G_neg, picos, '-')

    # bigram_labels
    for k, v in G_pos.items():
        if k not in bigram_labels:
            bigram_labels[k] = v
    
    if G_neg:
        for k,v in G_neg.items():
            if k not in bigram_labels:
                bigram_labels[k] = v  
            else:
                bigram_labels[k] = '!'+str(picos)

    return bigram_labels


def flatten(t):
    return [item for sublist in t for item in sublist]

def char_to_word_index(ci, sequence, tokens):
    """
    Given a character-level index (offset),
    return the index of the **word this char is in**
    """
    i = None
    for i, co in enumerate(tokens):
        if ci == co:
            return i
        elif ci < co:
            return i - 1
    return i

'''
Description:
    Converts label spans to token labels 
Args:
    matches (list): span matches 
    labels (list): labels generated by the labeling function
    terms (list): terms for which labels were generated by the labeling function
    start_spans (dict): dictionary where key is the start position of a span and value is the end position
    generated_labels (list): list initialized with 0's. 
    text_tokenized (list): tokenized training text
Returns:
    generated_labels (list):  modified 'generated_labels' list with P, I/C, O, S labels
'''
def spansToLabels(matches, df_data, picos:str):

    df_data_labels = []
    #Spans to labels
    for counter, match in enumerate(matches):

        abstain_lab = '!'+picos
        L = dict.fromkeys(range( len(list(df_data['offsets'])[counter]) ), abstain_lab) # initialize with abstain labels
        numerical_umls_labels = dict()


        if match:
            for (char_start, char_end), term, lab in match:
                
                # None labels are treated as abstains
                if not lab:
                    continue

                start, end = get_word_index_span(
                    (char_start, char_end - 1), list(df_data['text'])[counter], list(df_data['offsets'])[counter]
                )

                for i in range(start, end + 1):
                    L[i] = lab
        
        # Fetch numerical labels
        for k,v in L.items():
            numerical_umls_labels[k] = pico2labelMap[ v ]
        
        df_data_labels.append( numerical_umls_labels )

    return df_data_labels

'''
Description:
    Converts label spans to token labels for 'posPattern_i' heuristic LF
Args:
    matches (list): span matches 
    spans (list): spans corresponding to the matches
    labels (list): labels generated by the labeling function
    start_spans (dict): dictionary where key is the start position of a span and value is the end position
    generated_labels (list): list initialized with 0's. 
    text_tokenized (list): tokenized training text
Returns:
    generated_labels (list):  modified 'generated_labels' list with P, I/C, O, S labels
'''
def posPattern_i_to_labels(matches, spans, labels, start_spans, generated_labels, text_tokenized):

    for m, s, l in zip(matches, spans, labels):

        joined_m = ' '.join(m)
        if len( s ) == 1:
            joined_s = s[0]
        else:
            joined_s = ( s[0][0] , s[-1][-1] )

        start, end = get_word_index_span(
            (s[0][0], s[-1][-1] - 1), start_spans
        )
        if end and start:

            match_temp = ' '.join( [text_tokenized[x]  for x in range( start, end+1 )] )
            for x in range( start, end+1 ):
                if len( match_temp.strip() ) == len(joined_m.strip()):
                    #print( match_temp.strip() , ' ----- ',  joined_m.strip())
                    generated_labels[x] = l

    return generated_labels

'''
Description:
    Converts label spans to token labels for 'heurPattern_pa' heuristic LF
Args:
    matches (list): span matches 
    spans (list): spans corresponding to the matches
    labels (list): labels generated by the labeling function
    start_spans (dict): dictionary where key is the start position of a span and value is the end position
    generated_labels (list): list initialized with 0's. 
    text_tokenized (list): tokenized training text
Returns:
    generated_labels (list):  modified 'generated_labels' list with P, I/C, O, S labels
'''
def heurPattern_pa_to_labels(matches, labels, start_spans, generated_labels, text_tokenized):
    
    for m, l in zip(matches, labels):

        if len(m.group()) > 2:

            start, end = get_word_index_span(
                (m.span()[0], m.span()[1] - 1), start_spans
            )

            match_temp = ' '.join( [text_tokenized[x]  for x in range( start, end+1 )] )
            for x in range( start, end+1 ):
                if len( match_temp.strip() ) == len(m.group().strip()):
                    generated_labels[x] = l
                # else:
                #     print( match_temp.strip() , ' ----- ',  m.group().strip())

    return generated_labels
 
'''
Description:
    Fetch the label list and replace the labels P, I, O, S with corresponding numerical label
Args:
    l (list): input list with PIO labels
Returns:
    l_modified (list):  modified list with P, I, O, S replaced with corresponding numerical label
'''
def pico2label(l):

    l_modified = [ pico2labelMap[ l_i ] if l_i != 0 else l_i for l_i in l ]

    return l_modified

def write2File(labels_, tokens_, number_lfs, picos):
    
    indir = '/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation'
    
    df = pd.DataFrame( tokens_, columns=['tokens'] )
    
    for i in number_lfs:
        column_name = 'lf_' + str(i+1)
        df = df.merge( pd.DataFrame( labels_[i], columns=[column_name] ) , left_index=True, right_index=True )
    
    filename = 'lf_' + str(len(number_lfs)) + '.csv'
    with open(f'{indir}/{picos}/{filename}', 'w+') as wf:
        df.to_csv(f'{indir}/{picos}/{filename}', sep='\t')


def label_lf_partitions( partitioned_uml, umls_d, picos, text, token_flatten, spans, start_spans):

    accumulated_labels = []

    for i, lf in enumerate(partitioned_uml):

        print( 'Number of labeling functions: ', len(lf) )
        accumulated_lf = []
        number_lfs = []

        for number in range(0, len(lf)):

            sab_for_lf = lf[ number ]
            terms = flatten([ umls_d[ sab ]  for sab in  sab_for_lf])
            umls_labels = ontologyLF.OntologyLabelingFunction( text, token_flatten, spans, start_spans, terms, picos=None, fuzzy_match=False )
            accumulated_lf.append( umls_labels )
            number_lfs.append( number )

        accumulated_labels.append( accumulated_lf )
        write2File(accumulated_lf, token_flatten, number_lfs, picos)

def get_word_index_span(char_offsets, sequence, tokens):
    char_start, char_end = char_offsets
    return (char_to_word_index(char_start, sequence, tokens),
            char_to_word_index(char_end, sequence, tokens))

def get_label_dict(ont_list:list, picos: str) -> dict:

    ont_dict = dict()

    for l in ont_list:
        
        if l[0] not in ont_dict:
            ont_dict[ l[0] ] = l[1]
        elif l[0] in ont_dict and ont_dict[ l[0] ] != l[1]:
            abstain_label = '!' + picos
            ont_dict[ l[0] ] = abstain_label
        
    return ont_dict

'''
Description:
    Labels unlabeled training data with UMLS concepts
Args:
    outdir (str): directory path to store the weakly labeled candidates
    umls_d (dict): dictionary source with SAB (ontology) as key and (concepts, label or abstain) as value tuple
    picos (str): PICOS entity label
    text (str): training text
    token_flatten (list): flattened training tokens
    spans (list):  list of start and end positions for each token in text
    start_spans (dict): dictionary where keys are start positions and values are end positions of tokens in training text
    write (bool): switch to write training set to file
Returns:
    df (df): Dataframe with flattened tokens and corresponding weak labels
'''
def label_umls_and_write(outdir, umls_d, df_data, picos, arg_options, write):

    if str(outdir).split('/')[-1] == 'fuzzy':
        fuzzy_match = True
    else:
        fuzzy_match = False

    if arg_options.stop == True:
        sw_lf = sw
    elif arg_options.stop == False:
        sw_lf = None

    for k, v in umls_d.items():

        # convert ontology to dictionary
        labels = get_label_dict( v, picos )

        print( 'Fetching the labels for ', str(k) )
        if fuzzy_match == False:
            umls_labels = ontologyLF.OntologyLabelingFunctionX(  df_data['text'], df_data['tokens'], df_data['offsets'], labels, picos=picos, fuzzy_match=fuzzy_match, stopwords_general = sw_lf)
        else:
            umls_labels = ontologyLF.OntologyLabelingFunctionX(  df_data['text'], df_data['tokens'], df_data['offsets'], labels, picos=picos, fuzzy_match=fuzzy_match, stopwords_general = sw_lf)
        
        df_data_labels = spansToLabels(matches=umls_labels, df_data=df_data, picos=picos)

        assert len( df_data_labels ) == len( df_data['tokens'] ) == len( df_data['offsets'] )
        df_data['labels'] = df_data_labels
        count_all = 0
        for counter in df_data_labels:
            count_all = count_all + len( counter )

        if write == True:
            filename = str(picos) + '/lf_' + str(k) + '.tsv'
            print( filename , ' : ', count_all )
            df_data.to_csv(f'{outdir}/{filename}', sep='\t')
            # df_data.to_csv(f'{outdir}/{picos}/{filename}', sep='\t')


def listterms_to_dictterms(l:list, picos:str):

    return { l_i:picos for l_i in l } 

'''
Description:
    Labels unlabeled training data with input source. Input source could be non-UMLS concepts, ReGeX, DS source, hand-crafted dictionaries.
Args:
    outdir (str): directory path to store the weakly labeled candidates
    terms (list): list with either concepts/terms or regex for labeling 
    picos (str): PICOS entity label
    write (bool): switch to write training set to file
    ontology_name (str): String with ontology name for file writing
Returns:
    df (df): Dataframe with flattened tokens and corresponding weak labels
'''
def label_ont_and_write(outdir, terms, picos, df_data, write: bool, arg_options, ontology_name:str, extra_negs:list = None):

    if str(outdir).split('/')[-1] == 'fuzzy':
        fuzzy_match = True
    else:
        fuzzy_match = False

    if arg_options.stop == True:
        sw_lf = sw
    elif arg_options.stop == False:
        sw_lf = None

    labels = listterms_to_dictterms(terms, picos=picos)

    print( 'Fetching the labels for ', str(ontology_name) )
    if fuzzy_match == False:
        nonumls_labels = ontologyLF.OntologyLabelingFunctionX(  df_data['text'], df_data['tokens'], df_data['offsets'], labels, picos=picos, fuzzy_match=fuzzy_match, stopwords_general = sw_lf, extra_negs = extra_negs)
    elif fuzzy_match == True:
        nonumls_labels = ontologyLF.OntologyLabelingFunctionX(  df_data['text'], df_data['tokens'], df_data['offsets'], labels, picos=picos, fuzzy_match=fuzzy_match, stopwords_general = sw_lf, extra_negs = extra_negs)

    # convert labels to spans
    df_data_labels = spansToLabels(matches=nonumls_labels, df_data=df_data, picos=picos)

    assert len( df_data_labels ) == len( df_data['tokens'] ) == len( df_data['offsets'] )
    df_data['labels'] = df_data_labels
    count_all = 0
    for counter in df_data_labels:
        count_all = count_all + len( counter )

    if write == True:
        if extra_negs:
            filename = 'lf_' + str(ontology_name) + '_negs.tsv' 
        else:
            filename = 'lf_' + str(ontology_name) + '.tsv' 
        print( filename , ' : ', count_all )
        df_data.to_csv(f'{outdir}/{picos}/{filename}', sep='\t')
    else:
        return df_data

def label_regex_and_write(outdir, regex_compiled, picos, df_data, write: bool, arg_options, lf_name:str, neg_labs:list = None, neg_regex:list = None):

    if arg_options.stop == True:
        sw_lf = sw
    elif arg_options.stop == False:
        sw_lf = None

    regex_labels = ontologyLF.ReGeXLabelingFunction( df_data['text'], df_data['tokens'], df_data['offsets'], regex_compiled, picos=picos, stopwords_general=sw_lf, neg_labels=neg_labs, neg_regex=neg_regex )

    # convert labels to spans
    df_data_labels = spansToLabels(matches=regex_labels, df_data=df_data, picos=picos)
    count_all = 0
    for counter in df_data_labels:
        count_all = count_all + len( counter )


    assert len( df_data_labels ) == len( df_data['tokens'] ) == len( df_data['offsets'] )
    df_data['labels'] = df_data_labels

    if write == True:
        if neg_labs or neg_regex:
            filename = 'lf_' + str(lf_name) + '_negs.tsv'
        else:
            filename = 'lf_' + str(lf_name) + '.tsv'
        print( filename , ' : ', count_all )
        df_data.to_csv(f'{outdir}/{picos}/{filename}', sep='\t')
    else:
        return df_data


def label_heur_and_write( outdir, picos, df_data, write: bool, arg_options, lf_name: str, tune_for:str = None, neg_labs: list = None ):

    labels = []

    if arg_options.stop == True:
        sw_lf = sw
    elif arg_options.stop == False:
        sw_lf = None   

    if 'i_posreg' in lf_name:
        labels = heuristicLF.posPattern_i( df_data, picos = picos, stopwords_general=sw_lf, tune_for = tune_for )

    if 'lf_pa_regex_heur' in lf_name:
        labels = heuristicLF.heurPattern_pa( df_data, picos = picos, stopwords_general=sw_lf )

    if 'lf_ps_heurPattern_labels' in lf_name:
        labels = heuristicLF.heurPattern_p_sampsize( df_data, picos = picos, stopwords_general=sw_lf )

    if 'lf_o_heurpattern_labels' in lf_name:
        labels = heuristicLF.heurPattern_o_cal( df_data, picos = picos, stopwords_general=sw_lf )

    if 'lf_o2_heurpattern_labels' in lf_name:
        labels = heuristicLF.heurPattern_o_scale( df_data, picos = picos, stopwords_general=sw_lf )

    if 'lf_o3_heurpattern_labels' in lf_name:
        labels = heuristicLF.heurPattern_o_generic( df_data, picos = picos, stopwords_general=sw_lf )

    if 'lf_o4_heurpattern_labels' in lf_name:
        labels = heuristicLF.heurPattern_o_measurables( df_data, picos = picos, stopwords_general=sw_lf )

    if 'lf_o5_heurpattern_labels' in lf_name:
        labels = heuristicLF.heurPattern_o_passive_measure( df_data, picos = picos, stopwords_general=sw_lf )

    if 'lf_s_heurpattern_labels' in lf_name:
        labels = heuristicLF.heurPattern_s_cal( df_data, picos = picos, stopwords_general=sw_lf, tune_for = tune_for, neg_labs=neg_labs )

    # convert labels to spans
    df_data_labels = spansToLabels(matches=labels, df_data=df_data, picos=picos)
    count_all = 0
    for counter in df_data_labels:
        count_all = count_all + len( counter )

    assert len( df_data_labels ) == len( df_data['tokens'] ) == len( df_data['offsets'] )
    df_data['labels'] = df_data_labels
    count_all = 0
    for counter in df_data_labels:
        count_all = count_all + len( counter )

    if write == True:
        # filename = 'lf_' + str(lf_name) + '_negs.tsv'
        if neg_labs:
            filename = 'lf_' + str(lf_name) + '_negs.tsv'
        else:
            filename = 'lf_' + str(lf_name) + '.tsv'
        print( filename , ' : ', count_all )
        df_data.to_csv(f'{outdir}/{picos}/{filename}', sep='\t')
    else:
        return df_data



def label_abb_and_write(outdir, abb_source, picos, df_data, write: bool, arg_options, lf_name:str, extra_negs:list = None):


    if arg_options.stop == True:
        sw_lf = sw
    elif arg_options.stop == False:
        sw_lf = None

    labels = ontologyLF.AbbrevLabelingFunction( df_data, abb_source, picos = picos, stopwords_general=sw_lf, neg_labels = None )

    # convert labels to spans
    df_data_labels = spansToLabels(matches=labels, df_data=df_data, picos=picos)

    assert len( df_data_labels ) == len( df_data['tokens'] ) == len( df_data['offsets'] ) 
    df_data['labels'] = df_data_labels
    count_all = 0
    for counter in df_data_labels:
        count_all = count_all + len( counter )

    if write == True:
        if extra_negs:
            filename = 'lf_' + str(lf_name) + '_negs.tsv'
        else:
            filename = 'lf_' + str(lf_name) + '.tsv'
        print( filename , ' : ', count_all )
        df_data.to_csv(f'{outdir}/{picos}/{filename}', sep='\t')
    else:
        return df_data