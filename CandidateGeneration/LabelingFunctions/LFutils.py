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

'''
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
'''

def char_to_word_index(ci, sequence):
    """
    Given a character-level index (offset),
    return the index of the **word this char is in**
    """
    i = None
    if ci in sequence:
        i = sequence[ci]

    return i

def get_word_index_span(char_offsets, sequence):
    char_start, char_end = char_offsets
    return (char_to_word_index(char_start, sequence),
            char_to_word_index(char_end, sequence))


def spansToLabels(matches, labels, terms, start_spans, generated_labels, text_tokenized):

    for m, t, l in zip(matches, terms, labels):
        
        for m_i in m:

            if len(m_i.group()) > 2:

                start, end = get_word_index_span(
                    (m_i.span()[0], m_i.span()[1] - 1), start_spans
                )
                if end and start:

                    match_temp = ' '.join( [text_tokenized[x]  for x in range( start, end+1 )] )
                    for x in range( start, end+1 ):
                        if len( match_temp.strip() ) == len(t.strip()):
                            #print( match_temp.strip() , ' ----- ',  t.strip())
                            generated_labels[x] = l
                else:
                    pass
                    #print(start , ' : ', end , ' - ', t)

    return generated_labels


def heurspansToLabels(matches, spans, labels, start_spans, generated_labels, text_tokenized):

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