import spacy
import inflection
import re

NOUN = 'NOUN'
PROPN = 'PROPN'
ADJ = 'ADJ'

alphabet_regex = re.compile('[^a-zA-Z]')

valid_pos = {NOUN, PROPN, ADJ}
noun_pos = {NOUN, PROPN}

nlp = spacy.load('en')

def normalize_token(token):
    """
    Normalizes the given token.
    """
    word = token.lower_.strip().strip('=')
    if token.pos_ == NOUN:
        return inflection.singularize(word)
    else:
        return word

def is_valid_token(token):
    """
    Returns True if the token is valid
    (i.e. the token contains at least 2 alphabets)
    """
    return len(alphabet_regex.sub('', token.orth_)) >= 2

def is_meaningful_token(token):
    """
    Returns True if the token is meaningful for keyword extraction
    (i.e. it is a valid noun or adjective that is not a stop word)
    """
    return is_valid_token(token) and token.pos_ in valid_pos and not token.is_stop 

def is_noun(token):
    return token.pos_ in noun_pos