
import collections
import enum
import re
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from joblib import dump, load
from sanity_checks import *
from AnnotationAggregation.label_resolver import *
from itertools import groupby

def merge_sources(annotations):

    aggregated = dict()

    for source_key, source in annotations.items():

        if len(source.keys()) > 1:

            for target_key, target in source.items():

                if target_key != 'source':

                    if target_key not in aggregated:

                        aggregated[target_key] = target

                    else:
                        
                        for sentence_key, sentence in target.items():

                            if sentence_key not in aggregated[target_key]:

                                aggregated[target_key][sentence_key] = sentence

                            elif sentence_key in aggregated[target_key]:

                                for annot_key, annot in sentence.items():

                                    if annot_key not in aggregated[target_key][sentence_key] and 'tokens' not in annot_key:

                                        aggregated[target_key][sentence_key][annot_key] = annot

                                    elif annot_key in aggregated[target_key][sentence_key] and 'tokens' not in annot_key: # if the annotations need to be merged....

                                        # make new annotations by merging
                                        merged_annot = [0] * len(annot)
                                        for counter, (o_a, n_a) in enumerate(zip( aggregated[target_key][sentence_key][annot_key], annot  )):
                                            chosen_annot = max( o_a, n_a )
                                            merged_annot[counter] = chosen_annot

                                        aggregated[target_key][sentence_key][annot_key] = merged_annot

    return aggregated