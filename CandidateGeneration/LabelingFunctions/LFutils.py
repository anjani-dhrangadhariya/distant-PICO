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

def char_to_word_index(ci, sequence):
    """
    Given a character-level index (offset),
    return the index of the **word this char is in**
    """
    i = None
    for i, co in enumerate(sequence):
        if ci == co:
            return i
        elif ci < co:
            return i - 1
    return i

def get_word_index_span(char_offsets, sequence):
    char_start, char_end = char_offsets
    return (char_to_word_index(char_start, sequence),
            char_to_word_index(char_end, sequence))


def spansToLabels(matches, labels, terms, start_spans):

    generated_labels = [0] * len( start_spans )

    for m, t, l in zip(matches, terms, labels):
        
        for m_i in m:

            start, end = get_word_index_span(
                (m_i.span()[0], m_i.span()[1] - 1), start_spans
            )

            for x in range( start, end+1 ):
                generated_labels[x] = l

    return generated_labels