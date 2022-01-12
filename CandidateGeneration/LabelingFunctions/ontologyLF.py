def OntologyLabelingFunction(text, term, abstain_decision: bool = True, case_sensitive: bool = False):

    for i, t in enumerate(term):
        onto_term = t[0]
        for t_i in [onto_term, onto_term.lower(), onto_term.rstrip('s'), onto_term + 's']:
            if t_i in text:
                pass
            else:
                if len( t_i.split(' ') ) > 5:
                    print( t_i )