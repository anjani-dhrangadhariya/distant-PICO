from nltk import ngrams
import re

def OntologyLabelingFunction(text, term, max_ngram: int = 5, abstain_decision: bool = True, case_sensitive: bool = False):

    for i, t in enumerate(term):
        onto_term = t[0]

        if len( onto_term.split() ) > max_ngram:
            fivegrams = ngrams(onto_term.split(), max_ngram)
            # print( list(fivegrams) )

        for t_i in [onto_term, onto_term.lower(), onto_term.rstrip('s'), onto_term + 's']:
            if t_i in text:

                # Use matcher to get exact match and spans
                r = re.compile(t_i)
                spans = [[m.start(),m.end()] for m in r.finditer(text)]
                print( spans )

            
        if i == 10:
            break
