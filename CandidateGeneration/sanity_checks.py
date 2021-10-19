################################################################################
# Helper functions
################################################################################
def sanity_check(check_annots):

    for key, value in check_annots.items():
        for a_key, a_value in value.items():
            if 'source' not in a_key:
                for b_key, b_value in a_value.items():
                    assert len(b_value) == 2
                    assert len(b_value[0]) == len(b_value[1])

def sanity_check_intraagg(check_annots):

    for key, value in check_annots.items(): # each target
        for a_key, a_value in value.items(): # each sentence within the target
            if 'source' not in a_key: # if the key isnt source term
                assert len(a_value) == 2 # then assert if tokens and token annotations are both present in the aggregated annotations
                assert len(a_value[0]) == len(a_value[1]) # and check if both token and token annotation lengths are identical