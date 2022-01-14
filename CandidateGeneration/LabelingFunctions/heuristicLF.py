import re

def posPattern_i( text, validation_text_flatten, validation_pos_flatten, spans, picos: str, lr_window: int = 1 ):

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

            for i, pos_i in enumerate(reversed( validation_pos_flatten[ :index ] )):
                if pos_i in ['NN', 'NNS', 'NNP', 'NNS', 'JJ']:
                    longest_match = validation_text_flatten[ index-i : index+1 ] 
                    longest_match_span = spans[ index-i : index+1 ] 
                else:
                    break

            if len( longest_match ) > 1:
                regex_pos_matches.append( longest_match )
                regex_pos_spans.append( longest_match_span ) 
                labels.append( picos )              
    
    return regex_pos_matches, regex_pos_spans, labels