import re
import time

from nltk import ngrams
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize, word_tokenize

from LabelingFunctions import LFutils
from snorkel.labeling import labeling_function

'''
Description:
    Labeling function labels input data (str) with "Intervention" label using direct and fuzzy string matching
Args:
    text (str): input to be labeled
    text_tokenized (list): tokenized training text
    tokenized_spans (list): list of start and end positions for each token in text    
    tokenized_start_spans (dict): dictionary where key is the start position of a span and value is the end position
    source_terms (?): 
    picos (str): label to release for the input spans
    expand_term (bool): whether to expand the source term (plural, singular, lower)
    fuzzy_match (bool): switch for fuzzy bigram matching 
    max_ngram (int)
    abstain_decision (bool): switch to abstain or not
    case_sensitive (bool): switch for case sensitive matching
Returns:
    generated labels (list): 
'''
def OntologyLabelingFunction(text, 
                             text_tokenized,
                             tokenized_spans,
                             tokenized_start_spans,
                             source_terms,
                             picos: str,
                             expand_term: bool,
                             fuzzy_match: bool,
                             stopwords_general = list,
                             max_ngram: int = 5,
                             abstain_decision: bool = True, 
                             case_sensitive: bool = False):

    print( 'Total number of terms to check: ', len(source_terms) )

    ontology_matches = []
    label = []
    terms = []

    term_set = set()

    start_time = time.time()
    for i, term in enumerate(source_terms):

        t = term[0] if isinstance( term , tuple ) else term
        l = term[1] if isinstance( term , tuple ) else picos

        if '!' not in l: # abstain on abstain labels

            expandedTerms = LFutils.expandTerm( t , max_ngram, fuzzy_match) if expand_term else [t]      

            for t_i in expandedTerms:

                if isinstance( t_i , re.Pattern ):
                    if t_i.search(text.lower()):

                        matches = [m for m in t_i.finditer(text)]
                        term_set.add( t_i )
                        ontology_matches.append( matches )
                        terms.append( t_i )
                        label.append( l )
                else:
                    if t_i in text.lower():

                        r = re.compile(t_i)
                        matches = [m for m in r.finditer(text)]
                        ontology_matches.append( matches )
                        terms.append( t_i )
                        label.append( l )

    assert len(ontology_matches) == len(label)
    print( 'Before negative LF: ', len(ontology_matches) )

    # TODO: Use stopwords as Negative LF
    for sw in stopwords_general:
        r = re.compile(sw)
        matches = [m for m in r.finditer(text)]
        ontology_matches.append( matches )
        terms.append( t_i )
        neg_label = '-' + picos
        label.append( neg_label )
    assert len(ontology_matches) == len(label)
    print( 'After negative LF: ', len(ontology_matches) )

    generated_labels = len( text_tokenized ) * [0]
    generated_labels = LFutils.spansToLabels( ontology_matches, label, terms, tokenized_start_spans, generated_labels, text_tokenized )
    generated_labels = LFutils.pico2label( generated_labels )

    assert len( generated_labels ) == len( text_tokenized )

    print("--- %s seconds ---" % (time.time() - start_time))

    return generated_labels