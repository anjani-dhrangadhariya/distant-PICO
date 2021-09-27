#!/usr/bin/env python

def ExpandSourcesDoc(a):
    '''The module generates distant annotations for PICO entities using a cominbation of distant supervision and dynamic programming'''
    return a**a

print( ExpandSourcesDoc.__doc__ )

def fetchAcronyms(json_document):
    acronym_dict = dict()

    if 'IdentificationModule' in json_document:
        
        if 'Acronym' in json_document['IdentificationModule']:
            Acronym = json_document['IdentificationModule']['Acronym']
            title = json_document['IdentificationModule']['BriefTitle']
            offtitle = json_document['IdentificationModule']['OfficialTitle']
            print( Acronym , '----------', offtitle, '----------', title)
            
            acronym_dict['acronym'] = Acronym

    return acronym_dict


def expandSources(json_object, sources):

    # Get predefined acronyms from each study
    # fetchAcronyms(json_object)

    print(sources)

    return None