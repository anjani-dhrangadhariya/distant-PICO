import re

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
def posPattern_i( text, text_flatten, text_pos_flatten, spans, picos: str ):

    regex_matches = []

    for t_i in [  'treatment', 'therapy', 'surgery' ]:
        if t_i in text:
            r = re.compile(t_i)
            matches = [m for m in r.finditer(text)]
            regex_matches.extend( matches )

    regex_pos_matches = []
    regex_pos_spans = []
    labels = []

    for m in regex_matches:
        if m.span() in spans:
            index = spans.index( m.span() )

            longest_match = []
            longest_match_span = []

            for i, pos_i in enumerate(reversed( text_pos_flatten[ :index ] )):
                if pos_i in ['NN', 'NNS', 'NNP', 'NNS', 'JJ']:
                    longest_match = text_flatten[ index-i : index+1 ] 
                    longest_match_span = spans[ index-i : index+1 ] 
                else:
                    break

            if len( longest_match ) > 1:
                regex_pos_matches.append( longest_match )
                regex_pos_spans.append( longest_match_span ) 
                labels.append( picos )              
    
    return regex_pos_matches, regex_pos_spans, labels

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
def heurPattern_pa( text, text_flatten, text_pos_flatten, spans, picos: str ):

    regex_heur_matches = []
    regex_heur_labels = []


    eg_pattern = 'age.(.*?)(years|months)'

    compiled_pattern = re.compile( eg_pattern )

    if compiled_pattern.search(text):

        matches = [m for m in compiled_pattern.finditer(text)]

        for m in matches:
            if len(m[0]) < 20:
                regex_heur_matches.append( m )
                regex_heur_labels.append( picos )

    return regex_heur_matches, regex_heur_labels