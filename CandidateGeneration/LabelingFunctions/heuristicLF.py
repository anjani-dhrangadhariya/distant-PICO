import re

from LabelingFunctions.LFutils import heurspansToLabels, heurspansToLabels2, pico2label
from snorkel.labeling import labeling_function

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
def posPattern_i( text, text_tokenized, text_pos_flatten, spans, start_spans, picos: str ):

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
                    longest_match = text_tokenized[ index-i : index+1 ] 
                    longest_match_span = spans[ index-i : index+1 ] 
                else:
                    break

            if len( longest_match ) > 1:
                regex_pos_matches.append( longest_match )
                regex_pos_spans.append( longest_match_span ) 
                labels.append( picos )

    generated_labels = len( text_tokenized ) * [-1]
    generated_labels = heurspansToLabels(regex_pos_matches, regex_pos_spans, labels, start_spans, generated_labels, text_tokenized)
    generated_labels = pico2label( generated_labels )

    assert len(generated_labels) == len(text_tokenized)
    
    return generated_labels


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
def heurPattern_pa( text, text_tokenized, text_pos_flatten, spans, start_spans, picos: str ):

    regex_heur_matches = []
    regex_heur_labels = []


    eg_pattern = 'age.(.*?)(years|months)'

    compiled_pattern = re.compile( eg_pattern )

    if compiled_pattern.search(text):

        matches = [m for m in compiled_pattern.finditer(text)]

        for m in matches:
            if len(m[0]) < 30:
                regex_heur_matches.append( m )
                regex_heur_labels.append( picos )

    generated_labels = len(text_tokenized) * [-1]
    generated_labels = heurspansToLabels2(regex_heur_matches, regex_heur_labels, start_spans, generated_labels, text_tokenized)
    generated_labels = pico2label( generated_labels )

    return generated_labels


def heurPattern_p_sampsize( text, text_tokenized, text_pos_flatten, spans, start_spans, picos: str ):

    pattern = r' ?(patients?|subjects?|participants?|people?|individuals?|persons?|healthy individuals?|healthy adults?|children|toddlers?|adults?|healthy volunteers?|families?|men|women|teenagers?|families|parturients?|females?|males?)+'
    compiled_pattern = re.compile(pattern)

    for i, p in enumerate(text_pos_flatten):
        if i != len(text_pos_flatten)-2: # do not go to the last index
            if p == 'CD' and compiled_pattern.search( ' '.join(text_tokenized[i+1:i+3] ) ):
                
                matches = compiled_pattern.match( ' '.join(text_tokenized[i+1:i+3]) )
                
                if not matches:
                    #print( text_tokenized[i], ' '.join(text_tokenized[i+1:i+3]) )
                    continue
                else:
                    print( text_tokenized[i], matches.group() )

    return None