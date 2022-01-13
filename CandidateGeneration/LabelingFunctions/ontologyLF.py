from nltk import ngrams
import re
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize, word_tokenize
import time

def extractAnnotation(source, target, span_generator, match):
    
    token = list()
    annot = list()
    
    annotation_start_position = match[1][0]
    annotation_stop_position = match[1][0] + match[1][2]

    annotation = [0] * len(target)
    for n, i in enumerate(annotation):
        if n >= annotation_start_position and n <= annotation_stop_position:
            annotation[n] = 1

    for span in span_generator:
        token_ = target[span[0]:span[1]]

        annot_ = annotation[span[0]:span[1]]
        max_element_i = Counter(annot_)
        max_element = max_element_i.most_common(1)

        token.append(token_)
        annot.append(max_element)

    # Check if the number of annotations match number of tokens present in the sentence
    assert len(token) == len(annot)
        
    return token, annot

'''
Takes a labeling source (terms belonging to either one or more ontologies under a single LF arm).
'''
def OntologyLabelingFunction(text, text_tokenized, tokenized_spans, term, max_ngram: int = 5, abstain_decision: bool = True, case_sensitive: bool = False):

    print( 'Total number of terms to check: ', len(term) )

    spans_for_annot = []
    matched_term = []

    start_time = time.time()
    for i, t in enumerate(term):
        print(i)
        onto_term = t[0]

        #if len( onto_term.split() ) > max_ngram:
            #fivegrams = ngrams(onto_term.split(), max_ngram)
            # print( list(fivegrams) )

        for t_i in [onto_term, onto_term.lower(), onto_term.rstrip('s'), onto_term + 's']:
            if t_i in text:
                r = re.compile(t_i)
                matches = [m for m in r.finditer(text)]
                print('Number of matches identified {} for term {} : ', len(list(matches)) , t_i )


            
        if i == 50:
            break

    print("--- %s seconds ---" % (time.time() - start_time))

    print( 'Total spans identified: ', len(spans_for_annot) )
