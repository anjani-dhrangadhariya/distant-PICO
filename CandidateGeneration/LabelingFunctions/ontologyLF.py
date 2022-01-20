import re
import time

from nltk import ngrams
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize, word_tokenize

from LabelingFunctions import LFutils
from snorkel.labeling import labeling_function

'''
Takes a labeling source (terms belonging to either one or more ontologies under a single LF arm).
'''
def OntologyLabelingFunction(text, 
                             text_tokenized,
                             tokenized_spans,
                             tokenized_start_spans,
                             source_terms,
                             picos: str,
                             expand_term: bool,
                             fuzzy_match: bool,
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

        expandedTerms = LFutils.expandTerm( t , max_ngram, fuzzy_match) if expand_term else [t]      

        for t_i in expandedTerms:

            if isinstance( t_i , re._pattern_type ):
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

        # if i == 200:
        #     break

    assert len(ontology_matches) == len(label)

    generated_labels = len( text_tokenized ) * [-1]
    generated_labels = LFutils.spansToLabels( ontology_matches, label, terms, tokenized_start_spans, generated_labels, text_tokenized )
    generated_labels = LFutils.pico2label( generated_labels )

    assert len( generated_labels ) == len( text_tokenized )

    print("--- %s seconds ---" % (time.time() - start_time))

    return generated_labels