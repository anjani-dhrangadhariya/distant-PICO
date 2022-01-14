import re

def posPattern_i( text, validation_text_flatten, validation_pos_flatten, spans, lr_window: int = 1 ):

    regex_matches = []

    for t_i in [  'treatment', 'therapy', 'surgery' ]:

        if t_i in text:
            r = re.compile(t_i)
            matches = [m for m in r.finditer(text)]
            regex_matches.extend( matches )

    regex_pos_matches = []


    for m in regex_matches:
        if m.span() in spans:
            index = spans.index( m.span() )

            #print( validation_text_flatten[ index-lr_window-1 : index+1  ] )
            #print( validation_pos_flatten[ index-lr_window-1 : index+1  ] )

            for i, pos_i in enumerate(reversed( validation_pos_flatten[ :index ] )):
                if pos_i in ['NN', 'NNS', 'NNP', 'NNS', 'JJ']:
                    print( validation_text_flatten[ index-i : index+1 ] )
                else:
                    break


            
            '''
            if any( validation_pos_flatten[ index-2:index+2  ] ) in ['NN', 'NNS', 'NNP', 'NNS', 'JJ']:
                print( validation_pos_flatten[ index-2:index+2  ] )
            else:
                print( 'NO NO' )
            '''
                
    
    return None