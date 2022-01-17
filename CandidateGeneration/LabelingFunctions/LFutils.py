from nltk.tokenize import WhitespaceTokenizer, sent_tokenize, word_tokenize
from nltk import ngrams


def expandTerm( term , max_ngram, fuzzy_match):
    
    expandedTerm = []
    termVariations = []

    if len( term.split() ) > max_ngram:
        fivegrams = ngrams(term.split(), max_ngram)
        expandedTerm.extend( [' '.join(x)  for x in list(fivegrams)] )
    else:
        expandedTerm.extend( [term] )

    if fuzzy_match == True:
        bigrams = ngrams(term.split(), 2)
        expandedTerm.extend( [' '.join(x)  for x in list(bigrams)] )

    for eT in expandedTerm:
        termVariations.extend( [eT, eT.lower(), eT.rstrip('s'), eT + 's'] )

    return termVariations

def spans2Labels(matches, labels, spans):

    generateLabels = [0] * len( spans )

    for m,j in zip( matches, labels ):
        for m_i in m:
            match_span = m_i.span()
            if match_span in spans:
                pass
            else:
                print( m_i.group() )

    return None