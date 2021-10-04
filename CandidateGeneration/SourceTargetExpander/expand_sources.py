#!/usr/bin/env python

def ExpandSourcesDoc(a):
    '''The module generates distant annotations for PICO entities using a cominbation of distant supervision and dynamic programming'''
    return a**a

print( ExpandSourcesDoc.__doc__ )

# imports - general
import time

# imports - linguistic
import re
import spacy
from scispacy.abbreviation import AbbreviationDetector


nlp = spacy.load("en_core_sci_sm")

# Add the abbreviation detector to spacy pipeline
nlp.add_pipe("abbreviation_detector")

def fetchAcronyms(json_document):
    acronym_dict = dict()

    altered_tok = [tok.text for tok in doc]
    for abrv in doc._.abbreviations:
        altered_tok[abrv.start] = str(abrv._.long_form)

        print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form} \t  {value}")
        print( " ".join(altered_tok) )       

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

'''
Description:
    The funtion expands on the age terms of the study participants using a heuristic dictionary expansion

Args:
    dictionary value (string): free-text describing gender of the study pariticpants

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

    for eachStdAge in age_source['StdAge']:
        if 'Child' in eachStdAge:
            expanded_stdage.extend( baby_source )
            expanded_stdage.extend( child_source )
        elif 'Adult' in eachStdAge:
            expanded_stdage.extend( adult_source )
        elif 'Older Adult' in eachStdAge:
            expanded_stdage.extend( older_source )
    
    expanded_age_source['StdAge'] = expanded_stdage

    # Expand exact age
    if 'MinimumAge' in age_source and  'MaximumAge' in age_source:
        minage_num = age_source['MinimumAge'].split(' ')[0]
        minage_unit = age_source['MinimumAge'].split(' ')[1]

        maxage_num = age_source['MaximumAge'].split(' ')[0]
        maxage_unit = age_source['MaximumAge'].split(' ')[1]

        age_range_pattern =  '([Aa]ge[ds] )?(((\d{1,2}( years old)?(-| to | - | and )(.{1,3})?\d{1,2}) years) old)'
        compiled_pattern = re.compile(age_range_pattern)
        expanded_age_source['exactAge'] = compiled_pattern

    if 'MinimumAge' in age_source and 'MaximumAge' not in age_source:
        minage = age_source['MinimumAge']

        age_range_pattern = '([Aa]ge[ds] )?(\≥ |\> ||\< )?\d{1,2}( years (old)?( and above)?)'
        compiled_pattern = re.compile(age_range_pattern)
        expanded_age_source['exactAge'] = compiled_pattern

    if 'MaximumAge' not in age_source and 'MaximumAge' in age_source:
        # Usually this case never happens
        maxage_num = age_source['MaximumAge'].split(' ')[0]
        maxage_unit = age_source['MaximumAge'].split(' ')[1]

    return expanded_age_source


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

def getPOStags(value):

    doc = nlp(value)

    text = [ token.text for token in doc ]
    lemma = [ token.lemma_ for token in doc ]
    pos = [ token.pos_ for token in doc ]
    pos_fine = [ token.tag_ for token in doc ]
    depth = [ token.dep_ for token in doc ]
    shape = [ token.shape_ for token in doc ]
    is_alpha = [ token.is_alpha for token in doc ]
    is_stop = [ token.is_stop for token in doc ]

    pos_ed = [ value, text, lemma, pos, pos_fine ]

    return pos_ed

def appendPOSSED(expanded_dictionary, key, values):

    for value_i in values:
        possed_value = getPOStags(value_i)
        if key not in expanded_dictionary:
            expanded_dictionary[key] = [possed_value]
        else:
            expanded_dictionary[key].append( possed_value )

    return expanded_dictionary


def expandIntervention(intervention_source, fetch_pos = True, fetch_abb = True):

    expanded_intervention = dict()

    for key, value in intervention_source.items():
        if 'arm' not in key:
            
            if '&' in value and 'vs' not in value and ',' not in value  and ':' not in value and '(' not in value: # ampersand              
                values = value.split('&')
                values.append( value )
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, key, values)
                else:
                    expanded_intervention[key] = values

            elif '&' not in value and 'vs' in value and ',' not in value  and ':' not in value and '/' not in value and '(' not in value: # versus
                values = value.split('vs')
                values.append( value )
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, key, values)
                else:
                    expanded_intervention[key] = values

            elif '&' not in value and 'vs' not in value and ',' not in value  and ':' in value and '/' not in value and '(' not in value: # semi-colon
                values = value.split(':')
                values.append( value )
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, key, values)
                else:
                    expanded_intervention[key] = values

            elif '&' not in value and 'vs' not in value and ',' in value  and ':' not in value and '/' not in value and '(' not in value: # comma
                values = value.split(',')
                values.append( value )
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, key, values)
                else:
                    expanded_intervention[key] = values

            elif '&' not in value and 'vs' not in value and ',' not in value  and ':' not in value and '/' in value and '(' not in value: # forward slash
                values = value.split('/')
                values.append( value )
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, key, values)
                else:
                    expanded_intervention[key] = values

            else:
                if fetch_pos == True:
                    expanded_intervention = appendPOSSED(expanded_intervention, key, [value])
                else:
                    expanded_intervention[key] = values

        else:
            expanded_intervention[key] = value

    return expanded_intervention

def expandOutcomes(outcome_source, fetch_pos = True, fetch_abb = True):

    expanded_outcome = dict()

    for key, value in outcome_source.items():
        if fetch_pos == True:
            expanded_outcome = appendPOSSED(expanded_outcome, key, [value])
        else:
            expanded_outcome[key] = value

    print( expanded_outcome )

    return expanded_outcome


def expandSources(json_object, sources):

    expanded_sources = dict()

    # P
    expanded_gender = expandGender(sources['p_gender'])
    expanded_age = expandAge(sources['p_age'])
    # P - Condition needs abbreviation detection
    # P - Sample size does not need expansion

    # I/C
    expanded_intervention = expandIntervention(sources['i_name'], fetch_pos=True, fetch_abb=True)
    # I - synonyms do not require any expansion

    # O
    # Outcomes do not require other expansion except POS tagging
    expanded_prim_outcomes = expandOutcomes(sources['o_primary'])
    # expanded_second_outcomes = getPOStags(sources['o_secondary'])

    # S
    expanded_studytype = expandStudyType(sources['s_type'])

    expanded_sources['ep_gender'] = expanded_gender
    expanded_sources['ep_age'] = expanded_age

    expanded_sources['ei_name'] = expanded_intervention

    # expanded_sources['eo_primary'] = expanded_prim_outcomes
    # expanded_sources['eo_secondary'] = expanded_second_outcomes

    expanded_sources['es_type'] = expanded_studytype

    return expanded_sources