from cgitb import text
import json
from multiprocessing.sharedctypes import Value
from nis import match
import os
import re
import time
from CandGenUtilities.extract_shortlongforms import doc_term_forms

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
Returns:test_ebm_candidate_generation
'''
def OntologyLabelingFunctionX(corpus_text_series, 
                             corpus_words_series,
                             corpus_offsets_series,
                             source_terms: dict,
                             picos: str,
                             fuzzy_match: bool,
                             stopwords_general: list = None,
                             max_ngram: int = 5,
                             case_sensitive: bool = False,
                             longest_match_only = True,
                             extra_negs: list = None):


    # Add bigrams in case 
    if fuzzy_match == True:
        source_bigrams = LFutils.build_word_graph( source_terms, picos )
        source_terms.update(source_bigrams)

    # Add stopwords to the dictionary if 
    if isinstance(stopwords_general, list):
        for sw in stopwords_general:
            sw_abstain = '-'+picos
            if sw not in source_terms:
                source_terms[sw] = sw_abstain

    # Add Negative labels to the dictionary if they are not already defined as positive labels
    if isinstance(extra_negs, list):
        print( 'adding negative labels to the list' )
        for en in extra_negs:
            en_abstain = '-'+picos
            if en not in source_terms:
                source_terms[en] = en_abstain

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
                    if start != match:
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
                          regex_compiled,
                          picos: str,
                          stopwords_general:list = None,
                          neg_labels : list = None,
                          neg_regex = None
                          ):

    # Add stopwords to the lf (Negative labels)
    stop_dict = {}
    if isinstance(stopwords_general, list):
        stop_dict = { sw: '-'+picos for sw in stopwords_general }
    
    neg_dict = {}
    if isinstance(neg_labels, list):
        neg_dict = { nl: '-'+picos for nl in neg_labels }

    corpus_matches = []

    start_time = time.time()
    for words_series, offsets_series, texts_series in zip(corpus_words_series, corpus_offsets_series, corpus_text_series):

        matches = [] # append matches for a single sample

        if regex_compiled[0].search(texts_series.lower()): # emits +1

            match_result = [m for m in regex_compiled[0].finditer(texts_series)]

            for matches_i in match_result:
                matches.append(( [ matches_i.span()[0], matches_i.span()[1] ], matches_i.group(0), picos ))

        if neg_regex:
            for neg_regex_i in neg_regex:
                if neg_regex_i.search(texts_series.lower()): # emits -1
                    match_result = [m for m in neg_regex_i.finditer(texts_series)]

                    for matches_i in match_result:
                        matches.append(( [ matches_i.span()[0], matches_i.span()[1] ], matches_i.group(0), '-'+str(picos) ))


        #Find stopwrods here
        for k,v in stop_dict.items():
            match_indices = [i for i, x in enumerate(words_series) if x == k]

            for m_i in match_indices:
                matches.append(( [ offsets_series[m_i], offsets_series[m_i+1] ], k, v ))


        corpus_matches.append(matches)

    # Adds both the extra negative labels and also the stopwords
    if neg_labels:
        negative_labels_extra = OntologyLabelingFunctionX(  corpus_text_series, corpus_words_series, corpus_offsets_series, dict(), picos=picos, fuzzy_match=False, extra_negs = neg_labels)
        for counter, m in enumerate( negative_labels_extra ):
            corpus_matches[counter].extend( m )

    print("--- %s seconds ---" % (time.time() - start_time))

    return corpus_matches


def AbbreviationFetcher(df_data,
                          pos_dict,
                          neg_dict,
                          picos: str):

    # Abbreviations output dir
    filedict = { 'P': 'participant', 'I': 'intervention', 'O': 'outcome' }
    out_dir = f'/mnt/nas2/data/systematicReview/abbreviations/{filedict[picos]}/abb_sources.json'


    corpus_texts_series = df_data['text']
    corpus_words_series = df_data['tokens']
    corpus_pos_series = df_data['pos']
    corpus_offsets_series = df_data['offsets']

    meta_term_labels = {}

    for texts_series, words_series, pos_series, offsets_series in zip(corpus_texts_series, corpus_words_series, corpus_pos_series, corpus_offsets_series):
            
        term_labels_dictionary = doc_term_forms(words_series, pos_dict, neg_dict, pos_series, offsets_series, picos)
        meta_term_labels.update( term_labels_dictionary )    


    with open(out_dir, "a+") as outfile:
        #json.dump(meta_term_labels, outfile)
        outfile.write( json.dumps(meta_term_labels) )
        outfile.write( "\n" )


        # if os.stat(out_dir).st_size == 0:
        #     json.dump(meta_term_labels, outfile)
        # else:
        #     with open(out_dir, "r") as tf:
        #         temp_dict = json.load( tf )
        #         meta_term_labels.update( temp_dict ) 
        #         outfile.write( json.dumps(meta_term_labels) )
        #         outfile.write( '\n' )

    return meta_term_labels


def AbbrevLabelingFunction(df_data,
                          abb_sources,
                          picos: str,
                          stopwords_general:list = None,
                          max_ngram: int = 5,
                          case_sensitive: bool = False,
                          longest_match_only = True,
                          neg_labels:list = None):

    # Add stopwords to the lf (Negative labels)
    stop_dict = {}
    if isinstance(stopwords_general, list):
        stop_dict = { sw: '-'+picos for sw in stopwords_general }

    corpus_texts_series = df_data['text']
    corpus_words_series = df_data['tokens']
    corpus_pos_series = df_data['pos']
    corpus_offsets_series = df_data['offsets']

    start_time = time.time()

    corpus_matches = []
    longest_matches = []

    for texts_series, words_series, pos_series, offsets_series in zip(corpus_texts_series, corpus_words_series, corpus_pos_series, corpus_offsets_series):
            
        if len(stop_dict) > 0:
            abb_sources.update(stop_dict)

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
                    match_result = LFutils.match_term(term, abb_sources, case_sensitive)
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

    # Adds both the extra negative labels and also the stopwords
    if neg_labels:
        negative_labels_extra = OntologyLabelingFunctionX(  corpus_texts_series, corpus_words_series, corpus_offsets_series, dict(), picos=picos, fuzzy_match=False, extra_negs = neg_labels)
        for counter, m in enumerate( negative_labels_extra ):
            corpus_matches[counter].extend( m )
            longest_matches[counter].extend( m )

    print("--- %s seconds ---" % (time.time() - start_time))

    return longest_matches