def direct_alignment(target, source, PICOS, to_abstain=False):

    if target is not None :

        # Sentence tokenization
        collect_annotations = dict()
     
        annot = list()
        token = list()
        source = source.lower()
        eachSentence = target['text'].lower()
        eachSentence_tokens = target['tokens']
        eachSentence_pos = target['pos']
        eachSentence_posfine = target['pos_fine']

        if re.search( source, eachSentence ): # If there is at least one match 

            matches = re.finditer(source, eachSentence) # full direct match             
            for m in matches:
                token_i, annot_i = extractAnnotation(source, eachSentence, m, PICOS, isReGeX=True)
                if not annot and not token: # if the lists are empty
                    annot.extend( annot_i )
                    token.extend( token_i )
                else:
                    for counter, (o_a, n_a) in enumerate(zip( annot, annot_i )):
                        selected_annot = max( o_a, n_a )
                        annot[counter] = selected_annot

        elif len(source.split(' ')) > 1: # if finditer was not able to identify any direct matches
            # split the source into multiple tokens and identify if any of the "important tokens" is present in the target
            # Abstain from labeling it
            abstain_on_pos = nlp( source )
            for eachAbstain in abstain_on_pos:
                if eachAbstain.pos_  in ['NOUN', 'PROPN', 'ADJ' ] and re.search( str(eachAbstain.text) , eachSentence ):
                    inner_matches = re.finditer(str(eachAbstain.text), eachSentence) # full direct match    
                    for m in inner_matches:
                        token_i, annot_i = extractAnnotation(source, eachSentence, m, -1, isReGeX=True)
                        if not annot and not token: # if the lists are empty
                            annot.extend( annot_i )
                            token.extend( token_i )
                            print( annot )
                        else:
                            for counter, (o_a, n_a) in enumerate(zip( annot, annot_i )):
                                selected_annot = max( o_a, n_a )
                                annot[counter] = selected_annot
                else:
                    token_i, annot_i = extractAnnotation(source, eachSentence, [ 0, len(eachSentence) ], 0, isReGeX=False)
                    if not annot and not token: # if the lists are empty
                        annot.extend( annot_i )
                        token.extend( token_i )
                    else:
                        for counter, (o_a, n_a) in enumerate(zip( annot, annot_i )):
                            selected_annot = max( o_a, n_a )
                            annot[counter] = selected_annot
        else:
            token_i, annot_i = extractAnnotation(source, eachSentence, [ 0, len(eachSentence) ], 0, isReGeX=False)
            if not annot and not token: # if the lists are empty
                annot.extend( annot_i )
                token.extend( token_i )
            else:
                for counter, (o_a, n_a) in enumerate(zip( annot, annot_i )):
                    selected_annot = max( o_a, n_a )
                    annot[counter] = selected_annot



def extractAnnotation(source, target, match, PICOS, isReGeX):
    
    token = list()
    annot = list()
    
    span_generator = WhitespaceTokenizer().span_tokenize(target)

    if isReGeX == True:
        annotation_start_position = match.start()
        annotation_stop_position = match.end()
    # if isDifflib == True:
    #     annotation_start_position = match[1][0]
    #     annotation_stop_position = match[1][0] + match[1][2] # start + stop position
    else:
        annotation_start_position = match[0]
        annotation_stop_position = match[1]

    annotation = [0] * len(target)
    for n, i in enumerate(annotation):
        if n >= annotation_start_position and n <= annotation_stop_position: # if its anything between the start and the stop position of annotation
            annotation[n] = PICOS

    for span in span_generator:
        # span for each token is generated here
        token_ = target[span[0]:span[1]]
        
        annot_ = annotation[span[0]:span[1]]
        
        max_element_i = Counter(annot_)
        max_element = max_element_i.most_common(1)

        token.append(token_)
        annot.append(max_element[0][0])

    assert len(token) == len(annot)
       
    return token, annot