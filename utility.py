import spacy
import inflection

NOUN = 'NOUN'
PROPN = 'PROPN'
ADJ = 'ADJ'

valid_pos = {NOUN, PROPN, ADJ}

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

def is_meaningful_token(token):
    """
    Returns True if the token is meaningful for keyword extraction
    (i.e. the length of the token is >= 2 and the string is a noun or adjective.)
    """
    return len(token.orth_) >= 2 and not token.is_stop and token.pos_ in valid_pos