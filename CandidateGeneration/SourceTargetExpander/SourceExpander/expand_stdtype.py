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