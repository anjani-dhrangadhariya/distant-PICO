import re

from LabelingFunctions.LFutils import pico2label, posPattern_i_to_labels, heurPattern_pa_to_labels


'''
Description:
    Labeling function labels input data (str) with "Intervention" label using a heuristic combining ReGeX and part-of-speech (POS) tags

Args:
    text (str): input to be labeled
    text_flatten (list): list of tokens corresponding to the text input
    text_pos_flatten (list): list of POS tags corresponding to the token input
    spans (list): list of span values corresponding to the text and token inputs
    picos (str): label to release for the input spans

Returns:
    regex_pos_matches, regex_pos_spans, labels (lists): returns 3 lists each containing free text matches, spans and the span labels
'''
def posPattern_i ( df_data, picos: str, stopwords_general: list, tune_for: str = 'specificity' ):

    # Add stopwords to the lf (Negative labels)
    stop_dict = {}
    if stopwords_general:
        stop_dict = { sw: '-'+picos for sw in stopwords_general }

    corpus_text = df_data['text']
    corpus_tokens = df_data['tokens']
    corpus_offsets = df_data['offsets']
    corpus_pos = df_data['pos']

    regex_pos_corpus_matches = []

    for text, tokens, offsets, pos in zip( corpus_text, corpus_tokens, corpus_offsets, corpus_pos ):

        regex_matches = []

        # for t_i in [  'treatment', 'therapy', 'surgery' ]: # old 
        for t_i in [  'treatment', 'therapy', 'surgery', 'intervention', 'condition', 'group', 'arm', 'drug' ]: # new
            if t_i in text:
                r = re.compile(t_i)
                matches = [m for m in r.finditer(text)]
                regex_matches.extend( matches )

        regex_pos_matches = []

        for m in regex_matches:
            if m.span()[0] in offsets:
                index = offsets.index( m.span()[0] )

                longest_match = []
                longest_match_span = []

                for i, pos_i in enumerate(reversed( pos[ :index ] )):
                    if pos_i in ['NN', 'NNS', 'NNP', 'NNS', 'JJ']:
                        longest_match = tokens[ index-i : index+1 ] 
                        longest_match_span = offsets[ index-i : index+1 ] 
                    else:
                        break

                if tune_for == 'specificity':
                    if len( longest_match ) > 1:
                        regex_pos_matches.append( ([longest_match_span[0], longest_match_span[-1]], longest_match, picos) )
                else:
                    regex_pos_matches.append( ([longest_match_span[0], longest_match_span[0]], longest_match, picos) )

        # Match stopwords here
        for k,v in stop_dict.items():
            match_indices = [i for i, x in enumerate(tokens) if x == k]

            for m_i in match_indices:
                regex_matches.append(( [ offsets[m_i], offsets[m_i+1] ], k, v ))

        regex_pos_corpus_matches.append( regex_pos_matches )
    
    return regex_pos_corpus_matches


'''
Description:
    Labeling function labels input data (str) with "Participant: Age" label using a heuristic combining ReGeX and a rule

Args:
    text (str): input to be labeled
    text_flatten (list): list of tokens corresponding to the text input
    text_pos_flatten (list): list of POS tags corresponding to the token input
    spans (list): list of span values corresponding to the text and token inputs
    picos (str): label to release for the input spans

Returns:
    regex_heur_matches, regex_heur_labels (lists): returns 2 lists each containing free text matches (matching text and span) and the span labels
'''
def heurPattern_pa( df_data, picos: str, stopwords_general: list ):

    # Add stopwords to the lf (Negative labels)
    stop_dict = {}
    if stopwords_general:
        stop_dict = { sw: '-'+picos for sw in stopwords_general }

    regex_heur_matches = []

    eg_pattern = 'age.(.*?)(years|months)'
    compiled_pattern = re.compile( eg_pattern )

    corpus_text = df_data['text']
    corpus_tokens = df_data['tokens']
    corpus_offsets = df_data['offsets']
    corpus_pos = df_data['pos']

    regex_pos_corpus_matches = []

    for text, tokens, offsets, pos in zip( corpus_text, corpus_tokens, corpus_offsets, corpus_pos ):

        regex_pos_matches = []

        if compiled_pattern.search(text):

            matches = [m for m in compiled_pattern.finditer(text)]

            for m in matches:
                if len(m[0]) < 40:
                    regex_pos_matches.append( ([ m.span()[0], m.span()[1] ], m.group(0), picos) )

        # Match stopwords here
        for k,v in stop_dict.items():
            match_indices = [i for i, x in enumerate(tokens) if x == k]

            for m_i in match_indices:
                regex_pos_matches.append(( [ offsets[m_i], offsets[m_i+1] ], k, v ))

        regex_pos_corpus_matches.append( regex_pos_matches )

    return regex_pos_corpus_matches

'''
TODO: Development remains
'''
def heurPattern_p_sampsize( df_data, picos: str, stopwords_general: list ):

    # Add stopwords to the lf (Negative labels)
    stop_dict = {}
    if stopwords_general:
        stop_dict = { sw: '-'+picos for sw in stopwords_general }

    pattern = r' ?(patients?|subjects?|participants?|people?|individuals?|persons?|healthy individuals?|healthy adults?|children|toddlers?|adults?|healthy volunteers?|families?|men|women|teenagers?|families|parturients?|females?|males?)+'
    compiled_pattern = re.compile(pattern)

    corpus_text = df_data['text']
    corpus_tokens = df_data['tokens']
    corpus_offsets = df_data['offsets']
    corpus_pos = df_data['pos']

    regex_pos_corpus_matches = []

    for text, tokens, offsets, pos in zip( corpus_text, corpus_tokens, corpus_offsets, corpus_pos ):

        regex_pos_matches = []

        for i, p in enumerate(pos):

            if i != len(pos)-3: # do not go to the last index
                if p == 'CD' and compiled_pattern.search( ' '.join(tokens[i+1:i+3] ) ): # A number CD followed by pattern
                    
                    matches = compiled_pattern.match( ' '.join(tokens[i+1:i+3]) )

                    if not matches and pos[i+1] in ['NNP', 'JJ', 'VBZ', 'NN', 'NNS', 'JJR', 'VBG']:
                        term_match = ' '.join(tokens[i:i+3])
                        regex_pos_matches.append( ( [ offsets[i], offsets[i+3]  ], term_match, picos ) )

                    elif matches:
                        term_match = ' '.join(tokens[i:i+2])
                        regex_pos_matches.append( ( [ offsets[i], offsets[i+2]  ], term_match, picos ) )

        # Match stopwords here
        for k,v in stop_dict.items():
            match_indices = [i for i, x in enumerate(tokens) if x == k]

            for m_i in match_indices:
                regex_pos_matches.append(( [ offsets[m_i], offsets[m_i+1] ], k, v ))

        regex_pos_corpus_matches.append( regex_pos_matches )

    return regex_pos_corpus_matches