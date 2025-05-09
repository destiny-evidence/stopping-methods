import re
import logging
from nltk import WordNetLemmatizer, pos_tag, wordpunct_tokenize, sent_tokenize, word_tokenize
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn

logger = logging.getLogger('text-util')

lemmatizer = WordNetLemmatizer()
stopwords = sw.words('english')
NOALPH = re.compile(r'[^A-Za-z]+')


def lemmatize(token, tag):
    tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)
    return lemmatizer.lemmatize(token, tag)


def process_text_aggressive(text: str):
    return ' '.join([
        lemmatize(tok, tag)
        for sentence in sent_tokenize(text)
        for tok, tag in pos_tag(wordpunct_tokenize(sentence))
        if tok not in stopwords and len(NOALPH.sub('', tok)) >= 3
    ])


def process_text_light(text: str):
    return ' '.join([tok
                     for tok in word_tokenize(text)
                     if tok not in stopwords])
