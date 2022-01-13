import re
import time

from nltk import ngrams
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize, word_tokenize

from LabelingFunctions.LFutils import expandTerm

'''
Takes a labeling source (terms belonging to either one or more ontologies under a single LF arm).
'''
def OntologyLabelingFunction(text, 
                             text_tokenized,
                             tokenized_spans,
                             term,
                             fuzzy_match: bool = True,
                             max_ngram: int = 5,
                             abstain_decision: bool = True, 
                             case_sensitive: bool = False):

    print( 'Total number of terms to check: ', len(term) )

    ontology_matches = []

    start_time = time.time()
    for i, t in enumerate(term):

        onto_term = t[0]
        expandedTerms = expandTerm( t[0] , max_ngram, fuzzy_match)

        for t_i in expandedTerms:
            if t_i in text:
                r = re.compile(t_i)
                matches = [m for m in r.finditer(text)]
                ontology_matches.append( matches )

        if i == 50:
            break

    print("--- %s seconds ---" % (time.time() - start_time))

    return ontology_matches