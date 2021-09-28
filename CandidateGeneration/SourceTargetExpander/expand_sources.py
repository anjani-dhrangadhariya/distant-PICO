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

'''
Description:
    The funtion expands on the gender terms of the study participants using a heuristic dictionary expansion

Args:
    dictionary value (string): free-text describing gender of the study pariticpants

Returns:
    list: returns a list (in NER terms a dictionary) of expanded gender values according to gender of the study pariticpants
'''
def expandGender(gender_source):

    male_source = ['Male', 'Males', 'Men', 'Man', 'Boy', 'Boys']
    female_source = ['Female', 'Females', 'Women', 'Woman', 'Girl', 'Girls']

    expanded_gender_source = []

    if gender_source == 'All':
        expanded_gender_source.extend(female_source)
        expanded_gender_source.extend(male_source)
    elif gender_source == 'Female':
        expanded_gender_source.extend(female_source)
    elif gender_source == 'Male':
        expanded_gender_source.extend(male_source)

    print( expanded_gender_source )

    return expanded_gender_source


def expandSources(json_object, sources):

    expanded_sources = dict()

    # Get predefined acronyms from each study
    # fetchAcronyms(json_object)

    expanded_gender = expandGender(sources['p_gender'])
    

    return None