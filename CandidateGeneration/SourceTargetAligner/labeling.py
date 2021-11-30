from SourceTargetAligner.aligner import *

def scheme_ii( sources, expanded_targets, PICOS ):

    annotations = {}
    return annotations

def scheme_i( sources, expanded_targets, PICOS , toAbstain):

    annotations = {}

    for source_key, source_value in sources.items(): # Iterate each source

        for x, (eachSourceKey, eachSourceValue) in enumerate(source_value.items()): # Iterate each source

            for target_key, target_value in expanded_targets.items(): # now each value could be aligned on each target

                if target_key not in annotations:
                    annotations[target_key] = {}

                    for sentence_key, sentence_value in target_value.items():

                        picos_label = source_key.split('_')[0]
        #                 # print( picos_label, ' ------------------------------------ ', sentence_key )
                        tokens, annotations = direct_alignment(sentence_value, eachSourceValue, PICOS[picos_label], to_abstain = toAbstain[source_key] )



    # return annotations