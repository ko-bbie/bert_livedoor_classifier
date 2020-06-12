import re
import unicodedata


def category_dict():
    dic = {
        "sports-watch": [1, 0, 0, 0, 0, 0, 0, 0, 0],
        "smax": [0, 1, 0, 0, 0, 0, 0, 0, 0],
        "dokujo-tsushin": [0, 0, 1, 0, 0, 0, 0, 0, 0],
        "movie-enter": [0, 0, 0, 1, 0, 0, 0, 0, 0],
        "it-life-hack": [0, 0, 0, 0, 1, 0, 0, 0, 0],
        "kaden-channel": [0, 0, 0, 0, 0, 1, 0, 0, 0],
        "peachy": [0, 0, 0, 0, 0, 0, 1, 0, 0],
        "topic-news": [0, 0, 0, 0, 0, 0, 0, 1, 0],
        "livedoor-homme": [0, 0, 0, 0, 0, 0, 0, 0, 1],
    }
    return dic


def normalize(text):
    text = normalize_unicode(text)
    text = capitalize_symbols(text)
    return text


def normalize_unicode(text, form="NFKC"):
    normalized_text = unicodedata.normalize(form, text)
    normalized_text = re.sub(r"\s", " ", normalized_text)
    return normalized_text


def capitalize_symbols(text):
    # this function is often used for mecab morph analysis
    text = text.replace("・", "")
    table = str.maketrans(dict(zip("!?(),.:;/@%&[]", "！？（）、。：；／＠％＆［］")))
    capitalized_text = text.translate(table)
    return capitalized_text
