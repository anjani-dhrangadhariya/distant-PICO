# imports - general
import time

# imports - linguistic
import re
import spacy
from scispacy.abbreviation import AbbreviationDetector

from SourceTargetExpander.SourceExpander.expansion_utils import *

nlp = spacy.load("en_core_sci_sm")
# Add the abbreviation detector to spacy pipeline
nlp.add_pipe("abbreviation_detector")

'''
Description:
    The funtion expands on the gender terms of the study participants using a heuristic dictionary expansion

Args:
    dictionary value (string): free-text describing gender of the study pariticpants

Returns:
    list: returns a list (in NER terms a dictionary) of expanded gender values according to gender of the study pariticpants
'''
def expandGender(gender_source):

    male_source = ['Male', 'Males', ' Men ', ' Man ', 'Boy', 'Boys']
    female_source = ['Female', 'Females', 'Women', 'Woman', 'Girl', 'Girls']

    expanded_gender_source = {}

    if gender_source == 'All':
        gender_dict = []
        gender_dict.extend(female_source)
        gender_dict.extend(male_source)
        expanded_gender_source['dictionary'] = gender_dict

    elif gender_source == 'Female':
        expanded_gender_source['dictionary'] = female_source
    elif gender_source == 'Male':
        expanded_gender_source['dictionary'] = male_source

    return expanded_gender_source

'''
Description:
    The funtion expands the Participant age terms extracted from the clinical trial study using a heuristic dictionary expansion and ReGex

Args:
     dictionary (dict): dictionary describing age of the study pariticpants (exact age value and standardized age groups corresponding to MeSH)

Returns:
    dictionary: returns a dictionary of expanded age values and patterns according to age of the study pariticpants
'''
def expandAge(age_source):

    expanded_age_source = dict()
    expanded_stdage = []
    expanded_extAge = []

    # Expand standard age group
    baby_source = ['Newborn', 'Infant', 'Baby', 'Preterm', 'Neonate']
    child_source = ['Minors', 'Child', 'Adolescent', 'Young Adult', 'Young', 'Teen', 'Teenager', 'Teens', 'Teenagers']
    adult_source = ['Adult', 'Older Adult', 'Young Adult', 'Young']
    older_source = ['Older Adult', 'Aged', 'Elderly', 'Frail', 'Frail Older Adults', 'Frail Elders']

    if 'StdAge' in age_source:
        for eachStdAge in age_source['StdAge']:
            if 'Child' in eachStdAge:
                expanded_stdage.extend( baby_source )
                expanded_stdage.extend( child_source )
            elif 'Adult' in eachStdAge:
                expanded_stdage.extend( adult_source )
            elif 'Older Adult' in eachStdAge:
                expanded_stdage.extend( older_source )
    
    expanded_age_source['dictionary'] = expanded_stdage

    # Expand exact age pattern
    if 'MinimumAge' in age_source and 'MaximumAge' in age_source:
        minage_num = age_source['MinimumAge'].split(' ')[0]
        minage_unit = age_source['MinimumAge'].split(' ')[1]

        maxage_num = age_source['MaximumAge'].split(' ')[0]
        maxage_unit = age_source['MaximumAge'].split(' ')[1]

        age_range_pattern =  r'(([Aa]ge[ds]? )?\b([0-9]{1,2})\b(\s+years?|\s+months?)?(\s+old|-old)?\s?(-|to)\s?\b([0-9]{1,2})\b(\s+years?|\s+months?)+(\s+old|-old)?)'
        compiled_pattern = re.compile(age_range_pattern)
        expanded_age_source['regex'] = age_range_pattern

    if 'MinimumAge' in age_source and 'MaximumAge' not in age_source:
        minage = age_source['MinimumAge']

        age_range_pattern =  r'(([Aa]ge[ds]? ?)\b([0-9]{1,2})\b(\s+years?|\s+months?)?(\s+old|-old)?\s?(and above| and older)?)'
        compiled_pattern = re.compile(age_range_pattern)
        expanded_age_source['regex'] = age_range_pattern

    if 'MaximumAge' not in age_source and 'MaximumAge' in age_source:
        # Usually this case never happens
        maxage_num = age_source['MaximumAge'].split(' ')[0]
        maxage_unit = age_source['MaximumAge'].split(' ')[1]
        print('It is funny that only maximum age is mentioned in the clinical trials....')

    return expanded_age_source

'''
Description:
    The funtion expands the Participant condition mentions extracted from the clinical trial study by abbreviations using scispacy

Args:
    dictionary value (string): free-text describing study participants conditions
        fetch_pos (bool): True (default)
        fetch_abb (bool): True (default)

Returns:
    dictionary: returns a dictionary of expanded condition terms along with their abbreviations
'''
def expandCondition(condition_source, fetch_pos = False, fetch_abb = True):

    expanded_condition_source = {}

    for cond_i in condition_source:
        expanded_acronyms = fetchAcronyms( cond_i )

        if expanded_acronyms is not None:
            expanded_condition_source['dictionary'] = expanded_acronyms
            expanded_condition_source['dictionary'].append(cond_i) # also extend with the original value

        else:
            expanded_condition_source['dictionary'] = cond_i

    return expanded_condition_source

def expandSampleSize(sampsize_source):

    expanded_sampsize_source = {}


    for sampsize in sampsize_source:

        samp_size_pattern =  r'([0-9]+ ?(patients?|subjects?|participants?|people?|individuals?|persons?|healthy individuals?)+)'
        compiled_pattern = re.compile(samp_size_pattern)
        expanded_sampsize_source['regex'] = samp_size_pattern
        expanded_sampsize_source['dictionary'] = sampsize

    return expanded_sampsize_source