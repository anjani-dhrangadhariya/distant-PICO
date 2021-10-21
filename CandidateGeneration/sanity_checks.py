import collections

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


def getSecOrdKeys(myDict):
    return list( {j for i in myDict.values() for j in i} )


def getThirdOrdKeys(myDict): #don't use dict as  a variable name

    thirdorder = list( { k for i in myDict.values() for j in i.values() for k in j } )
    if 'tokens' in thirdorder:
        i = thirdorder.index("tokens")
        del thirdorder[i]
    else:
        thirdorder = thirdorder
    return thirdorder

def sanity_check_globalagg(temp_p, temp_ic, temp_o, temp_s, globalagg):

    first_order_keys =  list(set(list(temp_p.keys()) + list(temp_ic.keys()) + list(temp_o.keys()) +  list(temp_s.keys())))  
    assert collections.Counter( first_order_keys ) == collections.Counter( list(globalagg.keys()) )

    second_order_keys = list( set( getSecOrdKeys(temp_p) + getSecOrdKeys(temp_ic) + getSecOrdKeys(temp_o) + getSecOrdKeys(temp_s) )  )
    assert collections.Counter( second_order_keys ) == collections.Counter( list( set( getSecOrdKeys(globalagg))) )

    third_order_keys = getThirdOrdKeys(temp_p) + getThirdOrdKeys(temp_ic) + getThirdOrdKeys(temp_o) + getThirdOrdKeys(temp_s)
    assert collections.Counter(  set(third_order_keys) ) == collections.Counter( getThirdOrdKeys( globalagg ) )