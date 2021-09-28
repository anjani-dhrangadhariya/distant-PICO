#!/usr/bin/env python

def ExpandSourcesDoc(a):
    '''The module generates distant annotations for PICO entities using a cominbation of distant supervision and dynamic programming'''
    return a**a

print( ExpandSourcesDoc.__doc__ )

# imports
import re

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

    return expanded_gender_source

def expandAge(age_source):

    expanded_age_source = dict()

    # Expand standard age group
    baby_source = ['Newborn', 'Infant', 'Baby', 'Preterm', 'Neonate']
    child_source = ['Minors', 'Child', 'Adolescent', 'Young Adult', 'Young', 'Teen', 'Teenager', 'Teens', 'Teenagers']
    adult_source = ['Adult', 'Older Adult', 'Young Adult', 'Young']
    older_source = ['Older Adult', 'Aged', 'Elderly', 'Frail', 'Frail Older Adults', 'Frail Elders']

    print(age_source['StdAge'])

    # Expand the exact age
    
    return older_source

'''
Description:
    The funtion expands on the study type terms of the clinical trial study design using regulax expressions

Args:
    dictionary value (string): free-text describing study type of the trial design

Returns:
    string pattern: returns a ReGEx pattern ('re.Pattern') of expanded trial design values according to the trial design
        or a string: returns a 'N.A.' string if the trial design is not specified
'''
def expandStudyType(studytype_source):

    expanded_studytype_source = []

    ''' Expanded according the MeSH entry term (MeSH ID: D011897, ) from U.S. National Library of Medicine (NLM)'''
    # randomized_source = ['Random', 'Randomized', 'Randomised', 'Randomization', 'Randomisation']
    randomized_source_pattern = '([rR]andomi+[sz]e+d+)'

    ''' Expanded according the MeSH entry term (MeSH ID: D065228) from U.S. National Library of Medicine (NLM)'''
    # nonrandomized_source = ['Non-Random', 'Nonrandom', 'Non Random', 'Non-Randomized', 'Non-Randomised', 'Nonrandomized', 'Nonrandomised', 'Non Randomized', 'Non Randomised']
    nonrandomized_source_pattern = '([nN]o[nt][- ]+[rR]andomi+[sz]e+d+)'
    # XXX: Rather than a dictionary, expand it using a regular expression

    if studytype_source == 'N/A':
        return 'N.A.'
    elif studytype_source == 'Randomized':
        return re.compile(randomized_source_pattern)
    elif studytype_source == 'Non-Randomized':
        return re.compile(nonrandomized_source_pattern)

def expandSources(json_object, sources):

    expanded_sources = dict()

    # Get predefined acronyms from each study
    # fetchAcronyms(json_object)

    expanded_gender = expandGender(sources['p_gender'])
    # expanded_age = expandAge(sources['p_age'])
    expanded_studytype = expandStudyType(sources['s_type'])

    return None