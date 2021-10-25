import collections
import re
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from joblib import dump, load

def my_tokenizer(doc):
    tokens = doc.split(' ')
    return tokens


def text_prepare(text):
    REPLACE_BY_SPACE_RE = re.compile('[!@#$%^&*()[]{};:,./<>?\|`~=_+]')
    STOPWORDS = set(stopwords.words('english'))
    """
        text: a string
        
        return: modified initial string
    """
    text = REPLACE_BY_SPACE_RE.sub(' ', str(text))# replace REPLACE_BY_SPACE_RE symbols by space in text
    text = ' '.join([w for w in text.split() if not w in STOPWORDS])# delete stopwords from text
    return text.lower()