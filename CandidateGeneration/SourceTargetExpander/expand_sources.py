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

def expandIntervention(intervention_source):

    expanded_intervention = dict()

    for key, value in intervention_source.items():
        if 'arm' not in key:
            
            if '&' in value and 'vs' not in value and ',' not in value  and ':' not in value and '(' not in value: # ampersand
                values = value.split('&')
                value.extend( values )
                expanded_intervention[key] = values
            
            elif '&' not in value and 'vs' in value and ',' not in value  and ':' not in value and '(' not in value and '/' not in value: # versus
                values = value.split('vs')
                value.extend( values )
                expanded_intervention[key] = values

            elif '&' not in value and 'vs' not in value and ',' not in value  and ':' in value and '(' not in value and '/' not in value: # semi-colon
                values = value.split(':')
                value.extend( values )
                expanded_intervention[key] = values

            elif '&' not in value and 'vs' not in value and ',' in value  and ':' not in value and '(' not in value and '/' not in value: # comma
                values = value.split(',')
                value.extend( values )
                expanded_intervention[key] = value
                

            elif '&' not in value and 'vs' not in value and ',' not in value  and ':' not in value and '(' not in value and '/' in value: # forward slash
                values = value.split('/')
                value.extend( values )
                expanded_intervention[key] = value

            else:
                expanded_intervention[key] = value
        else:
            expanded_intervention[key] = value

    return expanded_intervention

def getPOStags(to_pos):

    pos_tagged = dict()

    for key, value in to_pos.items():

        doc = nlp(value)

        text = [ token.text for token in doc ]
        lemma = [ token.lemma_ for token in doc ]
        pos = [ token.pos_ for token in doc ]
        pos_fine = [ token.tag_ for token in doc ]
        depth = [ token.dep_ for token in doc ]
        shape = [ token.shape_ for token in doc ]
        is_alpha = [ token.is_alpha for token in doc ]
        is_stop = [ token.is_stop for token in doc ]

        pos_tagged[key] = [ value, text, lemma, pos, pos_fine ]
        
        altered_tok = [tok.text for tok in doc]
        for abrv in doc._.abbreviations:
            altered_tok[abrv.start] = str(abrv._.long_form)

            print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form} \t  {value}")
            print( " ".join(altered_tok) )       
        

        assert len(pos_tagged) == len(to_pos)

    return pos_tagged

def expandSources(json_object, sources):

    expanded_sources = dict()

    # P
    expanded_gender = expandGender(sources['p_gender'])
    expanded_age = expandAge(sources['p_age'])
    # P - Condition needs abbreviation detection
    # P - Sample size does not need expansion

    # I/C
    expanded_intervention = expandIntervention(sources['i_name'])
    # I - synonyms do not require any expansion

    # O
    # Outcomes do not require other expansion except POS tagging
    expanded_prim_outcomes = getPOStags(sources['o_primary'])
    expanded_second_outcomes = getPOStags(sources['o_secondary'])

    # S
    expanded_studytype = expandStudyType(sources['s_type'])

    expanded_sources['ep_gender'] = expanded_gender
    expanded_sources['ep_age'] = expanded_age

    expanded_sources['ei_name'] = expanded_intervention

    expanded_sources['eo_primary'] = expanded_prim_outcomes
    expanded_sources['eo_secondary'] = expanded_second_outcomes

    expanded_sources['es_type'] = expanded_studytype

    return expanded_sources