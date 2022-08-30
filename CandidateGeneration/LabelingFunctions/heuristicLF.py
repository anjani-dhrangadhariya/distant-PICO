# from ast import pattern
import re

from LabelingFunctions.LFutils import pico2label, posPattern_i_to_labels, heurPattern_pa_to_labels
from LabelingFunctions.ontologyLF import OntologyLabelingFunctionX

punct_list = [ '±', '+/-', '!', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '\[', '\\', '\]', '^', '_', '{', '|', '}', '~' ] 

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



def heurPattern_s_cal(df_data, picos: str, stopwords_general: list, tune_for: str = 'specificity', neg_labs: list = None ):

    # Add stopwords to the lf (Negative labels)
    stop_dict = {}
    if stopwords_general:
        stop_dict = { sw: '-'+picos for sw in stopwords_general }

    # Add specific negative labels for Study Type class
    # neg_dict = {}
    # if neg_labs:
    #     neg_dict = { nl: '-'+picos for nl in neg_labs }

    if tune_for == 'specificity':
        pattern = r'(((clinical|[cC]ontrolled)\s)+([tT]rial|[sS]tudy)+)'
    elif tune_for == 'sensitivity':
        pattern = r'(((clinical|[cC]ontrolled)\s)?([tT]rial|[sS]tudy)+)'

    compiled_pattern = re.compile(pattern)
    
    corpus_text = df_data['text']
    corpus_tokens = df_data['tokens']
    corpus_offsets = df_data['offsets']
    corpus_pos = df_data['pos']

    regex_pos_corpus_matches = []

    for text, tokens, offsets, pos in zip( corpus_text, corpus_tokens, corpus_offsets, corpus_pos ):

        regex_pos_matches = []

        for i, (t, p) in enumerate( zip(tokens, pos) ):
            
            if i != len(pos)-3: # do not go to the last index

                if compiled_pattern.match( ' '.join( tokens[ i:i+2 ] ) ): # if the pattern is found in the study
                    
                    # matched = compiled_pattern.findall( ' '.join( tokens[ i:i+2 ] ) )
                    matched = [m for m in compiled_pattern.finditer(' '.join( tokens[ i:i+2 ] ))]
                    orig_string =  ' '.join( tokens[ i:i+2 ] )

                    for matched_i in matched:

                        tokens_to_trace = list(reversed(tokens[ 0:i ]))
                        pos_to_trace = list(reversed(pos[ 0:i ]))
                        offs_to_trace = list(reversed(offsets[ 0:i ]))

                        back_search_agg = []
                        back_search_agg_o = []

                        back_search_agg.extend( list(reversed(tokens[ i:i+2 ]) ))
                        back_search_agg_o.extend( list(reversed(offsets[ i:i+2 ]) ))


                        for t_x, p_x, o_x in zip( tokens_to_trace, pos_to_trace, offs_to_trace ):

                            if p_x not in [ 'DT', 'IN', 'CC' ] and t_x not in [ '.', ')', '(', ':', '=', '>', '<' ]:
                                back_search_agg.append( t_x )
                                back_search_agg_o.append( o_x )
                            else:
                                break

                        tokens_app = list(reversed(back_search_agg))
                        spans = list(reversed(back_search_agg_o))

                        regex_pos_matches.append( ( [ spans[0], spans[-1]  ], ' '.join(tokens_app), picos ) )

        # Match stopwords here
        for k,v in stop_dict.items():
            match_indices = [i for i, x in enumerate(tokens) if x == k]

            for m_i in match_indices:
                regex_pos_matches.append(( [ offsets[m_i], offsets[m_i+1] ], k, v ))

        regex_pos_corpus_matches.append( regex_pos_matches )

     # Adds both the extra negative labels and also the stopwords
    if neg_labs:
        negative_labels_extra = OntologyLabelingFunctionX(  corpus_text, corpus_tokens, corpus_offsets, dict(), picos=picos, fuzzy_match=False, extra_negs = neg_labs)
        for counter, m in enumerate( negative_labels_extra ):
            regex_pos_corpus_matches[counter].extend( m )

    return regex_pos_corpus_matches



def heurPattern_o_cal( df_data, picos: str, stopwords_general: list, tune_for: str = 'specificity'  ):

    # Add stopwords to the lf (Negative labels)
    stop_dict = {}
    if stopwords_general:
        stop_dict = { sw: '-'+picos for sw in stopwords_general }

    pattern = r'([tT]otal|[aA]verage|[mM]ean|[mM]edian|[cC]omplete|[cC]umulative|[nN]on-cumulative|[pP]ostoperative)'
    compiled_pattern = re.compile(pattern)
    
    corpus_text = df_data['text']
    corpus_tokens = df_data['tokens']
    corpus_offsets = df_data['offsets']
    corpus_pos = df_data['pos']

    regex_pos_corpus_matches = []

    for text, tokens, offsets, pos in zip( corpus_text, corpus_tokens, corpus_offsets, corpus_pos ):

        regex_pos_matches = list()

        for counter, t in enumerate( tokens ):

            if compiled_pattern.search( t ): # if the pattern is found

                searched_pattern = compiled_pattern.findall( t )[0]

                matching_indices_i = []

                for i, (p, t_i) in enumerate( zip(pos[ counter: ], tokens) ): # then find the suceeding POS tags

                    if p in [ 'NN', 'NNS', 'NNP', 'NNP', 'JJ' ] and t_i not in punct_list: #  and if they are relevant
                        
                        if len( matching_indices_i ) == 0:
                            matching_indices_i.append( counter )
                        matching_indices_i.append( counter+i )
                    else:
                        break

                temp = list( set(matching_indices_i) )
                if len( temp ) > 2:

                    start_stop = list( set(matching_indices_i) )
                    term_match = ' '.join( tokens[start_stop[0] : start_stop[-1]] )
                    regex_pos_matches.append( ( [ offsets[start_stop[0]], offsets[start_stop[-1]]  ], term_match, picos ) )

        # Match stopwords here
        for k,v in stop_dict.items():
            match_indices = [i for i, x in enumerate(tokens) if x == k]

            for m_i in match_indices:
                regex_pos_matches.append(( [ offsets[m_i], offsets[m_i+1] ], k, v ))

        regex_pos_corpus_matches.append( regex_pos_matches )

    return regex_pos_corpus_matches

def heurPattern_o_scale( df_data, picos: str, stopwords_general: list, tune_for: str = 'specificity' ): #tune_for: str = 'specificity'

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

        pattern =  r'([sS]cales?|[sS]cores?|[qQ]uestionnaires?|[tT]ests?|\b[fF]orms?\b|[rR]ates?|[sS]ymptoms?|[lL]evels?|[vV]alues?)'
        r = re.compile(pattern)
        matches = [m for m in r.finditer(text)]
        regex_matches.extend( matches )

        regex_pos_matches = []

        for m in regex_matches:
            if m.span()[0] in offsets:
                index = offsets.index( m.span()[0] )

                longest_match = []
                longest_match_span = []

                for i, pos_i in enumerate(reversed( pos[ :index ] )):
                    if pos_i in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR' 'VB', 'VBD', 'VBZ', 'VBN', 'VBP', 'VBG', 'CD']:
                        longest_match = tokens[ index-i : index+1 ] 
                        longest_match_span = offsets[ index-i : index+1 ] 
                    else:
                        break

                if tune_for == 'specificity':
                    if len( longest_match ) > 1:
                        regex_pos_matches.append( ([longest_match_span[0], longest_match_span[-1]], longest_match, picos) )
                        # print( ([longest_match_span[0], longest_match_span[-1]], longest_match, picos) )
                else:
                    regex_pos_matches.append( ([longest_match_span[0], longest_match_span[0]], longest_match, picos) )

        # Match stopwords here
        for k,v in stop_dict.items():
            match_indices = [i for i, x in enumerate(tokens) if x == k]

            for m_i in match_indices:
                regex_matches.append(( [ offsets[m_i], offsets[m_i+1] ], k, v ))

        regex_pos_corpus_matches.append( regex_pos_matches )

    
    return regex_pos_corpus_matches

def heurPattern_o_generic( df_data, picos: str, stopwords_general: list, tune_for: str = 'specificity' ): #tune_for: str = 'specificity'

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

        pattern =  r'([bB]iomarkers?|[aA]activity|[eE]ndpoints?|[oO]utcomes?|[sS]cales?|[sS]cores?|[qQ]uestionnaires?|[tT]ests?|\b[fF]orms?\b|[rR]ates?|[sS]ymptoms?|[lL]evels?|[vV]alues?)'
        r = re.compile(pattern)
        matches = [m for m in r.finditer(text)]
        regex_matches.extend( matches )

        regex_pos_matches = []

        for m in regex_matches:
            if m.span()[0] in offsets:
                index = offsets.index( m.span()[0] )

                longest_match = []
                longest_match_span = []

                for i, pos_i in enumerate(reversed( pos[ :index ] )):
                    if pos_i in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR' 'VB', 'VBD', 'VBZ', 'VBN', 'VBP', 'VBG', 'CD']:
                        longest_match = tokens[ index-i : index+1 ] 
                        longest_match_span = offsets[ index-i : index+1 ] 
                    else:
                        break

                if tune_for == 'specificity':
                    if len( longest_match ) > 1:
                        regex_pos_matches.append( ([longest_match_span[0], longest_match_span[-1]], longest_match, picos) )
                        # print( ([longest_match_span[0], longest_match_span[-1]], longest_match, picos) )
                else:
                    regex_pos_matches.append( ([longest_match_span[0], longest_match_span[0]], longest_match, picos) )

        # Match stopwords here
        for k,v in stop_dict.items():
            match_indices = [i for i, x in enumerate(tokens) if x == k]

            for m_i in match_indices:
                regex_matches.append(( [ offsets[m_i], offsets[m_i+1] ], k, v ))

        regex_pos_corpus_matches.append( regex_pos_matches )

    
    return regex_pos_corpus_matches

def heurPattern_o_measurables( df_data, picos: str, stopwords_general: list, tune_for: str = 'specificity' ): #tune_for: str = 'specificity'

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

        pattern =  r'([cC]hanges?|[dD]ensity|[vV]olume|[mM]ass|[aA]ctivity|[eE]ffects?|[rR]esponses?|\b[rR]atio\b|[cC]oncentrations?|[nN]umber|[dD]urations?|[fF]unctions?|[tT]ime)'
        r = re.compile(pattern)
        matches = [m for m in r.finditer(text)]
        regex_matches.extend( matches )

        regex_pos_matches = []

        for m in regex_matches:
            if m.span()[0] in offsets:
                index = offsets.index( m.span()[0] )

                longest_match = []
                longest_match_span = []

                for i, pos_i in enumerate(reversed( pos[ :index ] )):
                    if pos_i in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR']:
                        longest_match = tokens[ index-i : index+1 ] 
                        longest_match_span = offsets[ index-i : index+1 ] 
                    else:
                        break

                if tune_for == 'specificity':
                    if len( longest_match ) > 1:
                        regex_pos_matches.append( ([longest_match_span[0], longest_match_span[-1]], longest_match, picos) )
                        # print( ([longest_match_span[0], longest_match_span[-1]], longest_match, picos) )
                else:
                    regex_pos_matches.append( ([longest_match_span[0], longest_match_span[0]], longest_match, picos) )

        # Match stopwords here
        for k,v in stop_dict.items():
            match_indices = [i for i, x in enumerate(tokens) if x == k]

            for m_i in match_indices:
                regex_matches.append(( [ offsets[m_i], offsets[m_i+1] ], k, v ))

        regex_pos_corpus_matches.append( regex_pos_matches )

    
    return regex_pos_corpus_matches

def heurPattern_o_passive_measure( df_data, picos: str, stopwords_general: list, tune_for: str = 'specificity' ): #tune_for: str = 'specificity'

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

        pattern =  r'([nN]umbers?|[cC]hanges?|[vV]olume|\b[rR]atio\b|[rR]esponses?|[nN]umber|[cC]oncentrations?|[dD]urations?|[fF]unctions?|[tT]ime)'
        r = re.compile(pattern)
        matches = [m for m in r.finditer(text)]
        regex_matches.extend( matches )

        regex_pos_matches = []

        for m in regex_matches:
            if m.span()[0] in offsets:
                index = offsets.index( m.span()[0] )

                longest_match_b = []
                longest_match_span_b = []

                longest_match_f = []
                longest_match_span_f = []

                # backward search 
                for i, pos_i in enumerate(reversed( pos[ :index ] )):
                    if pos_i in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR']:
                        longest_match_b = tokens[ index-i : index+1 ] 
                        longest_match_span_b = offsets[ index-i : index+1 ] 
                    else:
                        break

                if tune_for == 'specificity':
                    if len( longest_match_b ) > 1:
                        regex_pos_matches.append( ([longest_match_span_b[0], longest_match_span_b[-1]], longest_match_b, picos) )
                        # print( ([longest_match_span_b[0], longest_match_span_b[-1]], longest_match_b, picos) )
                else:
                    regex_pos_matches.append( ([longest_match_span_b[0], longest_match_span_b[0]], longest_match_b, picos) )

                # forward search 
                for i, pos_i in enumerate( ( pos[ index: ] ) ):
                    if pos_i in ['IN', 'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR']:
                        longest_match_f = tokens[ index : index+i ] 
                        longest_match_span_f = offsets[ index : index+i ] 
                    else:
                        break

                if tune_for == 'specificity':
                    if len( longest_match_f ) > 2:
                        regex_pos_matches.append( ([longest_match_span_f[0], longest_match_span_f[-1]], longest_match_f, picos) )
                        # print( ([longest_match_span_f[0], longest_match_span_f[-1]], longest_match_f, picos) )
                else:
                    regex_pos_matches.append( ([longest_match_span_f[0], longest_match_span_f[0]], longest_match_f, picos) )

        # Match stopwords here
        for k,v in stop_dict.items():
            match_indices = [i for i, x in enumerate(tokens) if x == k]

            for m_i in match_indices:
                regex_matches.append(( [ offsets[m_i], offsets[m_i+1] ], k, v ))

        regex_pos_corpus_matches.append( regex_pos_matches )

    
    return regex_pos_corpus_matches