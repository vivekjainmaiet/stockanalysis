chars_to_remove= '€$!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'

def lower(text):

    return text.lower()


def clean_text(text):

    for punctuation in chars_to_remove:
        text = text.replace(punctuation, ' ')

    return text
