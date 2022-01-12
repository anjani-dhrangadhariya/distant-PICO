from nltk import ngrams


def OntologyLabelingFunction(text, term, max_ngram: int = 5, abstain_decision: bool = True, case_sensitive: bool = False):

    for i, t in enumerate(term):
        onto_term = t[0]

        if len( onto_term.split() ) > max_ngram:
            fivegrams = ngrams(onto_term, max_ngram)
            print( list(fivegrams) )

        for t_i in [onto_term, onto_term.lower(), onto_term.rstrip('s'), onto_term + 's']:
            if t_i in text:
                pass
            # else:
            #     if len( t_i.split(' ') ) > 5:
            #         print( t_i )