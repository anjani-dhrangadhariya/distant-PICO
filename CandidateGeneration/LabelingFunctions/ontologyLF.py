from cgitb import text
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
    fuzzy_match (bool): switch for fuzzy bigram matching 
    max_ngram (int)
    case_sensitive (bool): switch for case sensitive matching
Returns:
    generated labels (list): 
'''
def OntologyLabelingFunctionX(corpus_text_series, 
                             corpus_words_series,
                             corpus_offsets_series,
                             source_terms: dict,
                             picos: str,
                             fuzzy_match: bool,
                             stopwords_general = list,
                             max_ngram: int = 5,
                             case_sensitive: bool = False,
                             longest_match_only = True):

    # Add stopwords to the dictionary if 
    if stopwords_general:
        for sw in stopwords_general:
            sw_abstain = '!'+picos
            source_terms[sw] = sw_abstain

    corpus_matches = []
    longest_matches = []

    for words_series, offsets_series, texts_series in zip(corpus_words_series, corpus_offsets_series, corpus_text_series):

        matches = []

        for i in range(0, len(words_series)):

            match = None
            start = offsets_series[i]

            for j in range(i + 1, min(i + max_ngram + 1, len(words_series) + 1)):
                end = offsets_series[j - 1] + len(words_series[j - 1])

                # term types: normalize whitespace & tokenized + whitespace
                for term in [
                    re.sub(r'''\s{2,}''', ' ', texts_series[start:end]).strip(),
                    ' '.join([w for w in words_series[i:j] if w.strip()])
                ]:
                    match_result = LFutils.match_term(term, source_terms, case_sensitive)
                    if match_result[0] == True:
                        match = end
                        break

                if match:
                    term = re.sub(r'''\s{2,}''', ' ', texts_series[start:match]).strip()
                    matches.append(( [start, match], term, match_result[-1] ))

        corpus_matches.append( matches )

        if longest_match_only:
            # sort on length then end char
            matches = sorted(matches, key=lambda x: x[0][-1], reverse=1)
            f_matches = []
            curr = None
            for m in matches:
                if curr is None:
                    curr = m
                    continue
                (i, j), _, _ = m
                if (i >= curr[0][0] and i <= curr[0][1]) and (j >= curr[0][0] and j <= curr[0][1]):
                    pass
                else:
                    f_matches.append(curr)
                    curr = m
            if curr:
                f_matches.append(curr)

            # if f_matches:
            longest_matches.append( f_matches )

    if longest_match_only:
        assert len( longest_matches ) == len(corpus_words_series)
        return longest_matches
    else:
        assert len( corpus_matches ) == len(corpus_words_series)
        return corpus_matches


def ReGeXLabelingFunction(corpus_text_series, 
                          corpus_words_series,
                          corpus_offsets_series,
                          source_terms,
                          picos: str
                          ):

    print( 'Total number of terms to check: ', len(source_terms) )

    corpus_matches = []

    start_time = time.time()
    for i, term in enumerate(source_terms):

        matches = []

        for words_series, offsets_series, texts_series in zip(corpus_words_series, corpus_offsets_series, corpus_text_series):

            if term.search(texts_series.lower()):

                match_result = [m for m in term.finditer(texts_series)]

                for matches_i in match_result:
                    matches.append(( [ matches_i.span()[0], matches_i.span()[1] ], matches_i.group(0), picos ))

            corpus_matches.append(matches)

    print("--- %s seconds ---" % (time.time() - start_time))


    return corpus_matches