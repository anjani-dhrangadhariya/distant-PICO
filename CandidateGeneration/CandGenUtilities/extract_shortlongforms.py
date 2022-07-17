"""
Labeling function implementation of  "A simple algorithm for identifying
abbreviation definitions in biomedical text"
Schwartz AS, Hearst MA
Computer Science Division, University of California,
Berkeley, Berkeley, CA 94720, USA. sariel@cs.berkeley.edu

Pac Symp Biocomput. 2003;:451-62.

http://psb.stanford.edu/psb-online/proceedings/psb03/abstracts/p451.html

TODO: Refactor

"""
import re
import collections
from typing import List, Set, Dict

def is_short_form(text, min_length=2):
    """ Rule-based function for determining if a token is likely
    an abbreviation, acronym or other "short form" mention

    Parameters
    ----------
    text
    min_length

    Returns
    -------

    """
    accept_rgx = '[0-9A-Z-]{2,8}[s]*'
    reject_rgx = '([0-9]+/[0-9]+|[0-9]+[-][0-7]+)'

    keep = re.search(accept_rgx, text) is not None
    keep &= re.search(reject_rgx, text) is None
    keep &= not text.strip("-").isdigit()
    keep &= "," not in text
    keep &= len(text) < 15

    # reject if too short too short or contains lowercase single letters
    # reject = (len(text) > 3 and not keep and any(c.isalpha() for c in text) == False)
    reject = (len(text) > 3 and not keep)
    reject |= (len(text) <= 3 and re.search("[/,+0-9-]", text) is not None)
    reject |= (len(text) < min_length)
    reject |= (len(text) <= min_length and text.islower())

    return False if reject else True


def get_parenthetical_short_forms(tokens):
    """Generator that returns indices of all words directly
    wrapped by parentheses or brackets.

    Parameters
    ----------
    sentence

    Returns
    -------

    """
    for i, _ in enumerate(tokens):
        if i > 0 and i < len(tokens) - 1: # any token except first and last
            window = tokens[i - 1:i + 2] # Why +2? token 0 = opening bracket, token 1 = short form, token 2 = closing bracket
            if window[0] == "(" and window[-1] == ")":
                if is_short_form(window[1]):
                    yield i


def extract_long_form(i, tokens, pos, token_offsets, max_dup_chars=2):
    '''
    Search the left window for a candidate long-form sequence.
    Use the heuristic of "match first character" to guess long form
    '''

    token_indices = [i for i, w in enumerate(tokens)]
    short_form = tokens[i]
    left_window = [w for w in tokens[0:i]] # all tokens in the left
    left_window_indices = [i for i, w in enumerate(tokens[0:i])] # all token indices in the left

    # strip brackets/parentheses
    while left_window and left_window[-1] in ["(", "[", ":"]:
        left_window.pop()
        left_window_indices.pop()

    if len(left_window) == 0: # if nothing remains in the left window
        return None

    # match longest seq to the left of our short form
    # that matches on starting character
    long_form = []
    long_form_indices = []
    char = short_form[0].lower()
    letters = [t[0].lower() for t in short_form]
    letters = [t for t in letters if t == char]
    letters = letters[0:min(len(letters), max_dup_chars)]

    matched = False

    for indice, t in zip( left_window_indices[::-1],left_window[::-1] ):
        if t:
            if t[0] in "()[]-+,":
                break

            if len(letters) == 1 and t[0].lower() == letters[0]:
                long_form += [t]
                long_form_indices += [str(indice)]
                matched = True
                break

            elif len(letters) > 1 and t[0].lower() == letters[0]:
                long_form += [t]
                long_form_indices += [str(indice)]
                matched = True
                letters.pop(0)

            else:
                long_form += [t]
                long_form_indices += [str(indice)]


    # We didn't find the first letter of our short form, so
    # back-off and choose the longest contiguous noun phrase
    if (len(left_window) == len(long_form) and \
        letters[0] != t[0].lower() and \
        len(long_form[::-1]) > 1) or not matched:

        tags = list(zip(tokens[0:i - 1], pos[0:i - 1], token_indices[0:i - 1]))[::-1]
        noun_phrase = []

        while tags:
            t = tags.pop(0)
            if re.search("^(NN[PS]*|JJ)$", t[1]):
                noun_phrase.append(t)
            else:
                break

        if noun_phrase:
            long_form = list( zip(*noun_phrase) )[0]
            long_form_indices = list( zip(*noun_phrase) )[-1]

    # print( short_form , ' : ', long_form, ' : ', long_form_indices )

    # # create candidate
    # n = len(long_form[::-1])
    # offsets = token_offsets[0:i - 1][-n:]
    # char_start = min(offsets)
    # words = tokens[0:i - 1][-n:]

    # offsets = map(lambda x: len(x[0]) + x[1], zip(words, offsets))
    # char_end = max(offsets)

    return short_form, list(reversed(long_form)), list(reversed(long_form_indices))


def get_short_form_index(cand_set):
    '''
    Build a short_form->long_form mapping for each document. Any
    short form (abbreviation, acronym, etc) that appears in parenthetical
    form is considered a "definition" and added to the index. These candidates
    are then used to augment the features of future mentions with the same
    surface form.
    '''

    sf_index = {}
    for doc in cand_set:

        for sent in doc.sentences:
            for i in get_parenthetical_short_forms(sent):
                short_form = sent.words[i]
                long_form_cand = extract_long_form(i, sent)

                if not long_form_cand:
                    continue
                if doc.doc_id not in sf_index:
                    sf_index[doc.doc_id] = {}
                if short_form not in sf_index[doc.doc_id]:
                    sf_index[doc.doc_id][short_form] = []
                sf_index[doc.doc_id][short_form] += [long_form_cand]

    return sf_index



def doc_term_forms(words_tokens, pos_dict, neg_dict, pos, offsets, picos, stopwords):

    abbrv_map = collections.defaultdict(list)

    for i in get_parenthetical_short_forms(words_tokens):
        short_form = words_tokens[i]
        
        short_form, long_form, long_form_indices = extract_long_form(i, words_tokens, pos, token_offsets=offsets)
        # print(  short_form, long_form, long_form_indices )

        if not long_form:
            continue

        abbrv_map[short_form].append( ( long_form, long_form_indices ) )

    # map each short form to a class label
    term_labels = {}
    for sf in abbrv_map:
        # print(sf)
        # label = None
        # print( sf,  abbrv_map[sf] )
        for eachLf in abbrv_map[sf]:

            longform_text = ' '.join(eachLf[0])
            
            if longform_text in pos_dict or longform_text.lower() in pos_dict:
                term_labels[longform_text] = picos
                term_labels[sf] = picos

            if longform_text in neg_dict or longform_text.lower() in neg_dict:
                term_labels[longform_text] = str('-') + picos
                term_labels[sf] = str('-') + picos

            # for eachLf in abbrv_map[sf]:
            #     if term.text in pos_dict:
            #         label = label
            #     elif term.text.lower() in neg_dict:
            #         label = label
            #         break

    #     # if any long form is in our class dictionaries,
    #     # treat this as a synset for the class label
    #     if label:
    #         term_labels[sf] = label
    #         for term in abbrv_map[sf]:
    #             term_labels[term.text.lower()] = label


    return term_labels