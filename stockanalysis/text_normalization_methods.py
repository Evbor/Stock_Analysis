import re
import lxml
import unicodedata
import en_core_web_sm

from bs4 import BeautifulSoup

# Instantiating Spacy NLP object with the parser and named entity recognition components
nlp = en_core_web_sm.load(disable=['parser', 'ner'])

# Defining Functions and Classes

def extract_8k(doc):
    eight_k = re.findall('<DOCUMENT>\n<TYPE>8-K.*?<SEQUENCE>1(.*?)</DOCUMENT>', doc, re.DOTALL | re.IGNORECASE)
    text = re.findall('<TEXT>(.*?)</TEXT>', eight_k[0], re.DOTALL | re.IGNORECASE)
    return text

def extract_html(doc):
    html = re.findall('<HTML(?:>| .*?>).*?</HTML>', doc, re.DOTALL | re.IGNORECASE)
    return html

def strip_tags(doc):
    # extracting 8-K <DOCUMENT> tag from the filing
    eight_k = extract_8k(doc)
    assert len(eight_k) == 1, 'Check re for 8-K extraction, either multiple 8-K DOCUMENT tags or bad re'

    # extracting <html> tag if any
    html = extract_html(eight_k[0])
    assert 0 <= len(html) <= 1, 'Check re for extracting html tags'

    # if html exists
    if len(html) == 1:
        html = html[0]
        soup = BeautifulSoup(html, 'lxml')
        stripped = soup.get_text()
    else:
        soup = BeautifulSoup(eight_k[0], 'lxml')
        stripped = soup.get_text()

    return stripped

def strip_accented_chars(doc):
    doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return doc

def strip_special_chars(doc):
    doc = re.sub('[^$A-Za-z0-9%\s.\']', '', doc)
    return doc

def lemmatize(doc):
    document = nlp(doc)
    doc = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in document])
    return doc

def strip_stop_words(doc):
    document = nlp(doc)
    doc = ' '.join([word.text for word in document if not word.is_stop])
    return doc

def strip_large_words(doc, cut_off=20):
    return ' '.join([word for word in doc.split() if len(word) <= cut_off])

def normalize_document(doc, tags_strip=True, accent_char_strip=True, lower_case=True,
                       no_newlines=True, special_char_strip=True, space_nums=True,
                       lemmatize_words=True, remove_stop_words=True, strip_extra_spaces=True,
                       remove_large_words=False, debug=False):
    '''
    Preprocesses the document :param doc: and returns the normalized document.

    :param doc: string, document to normalize
    :param xml_strip: bool, set to True to strip the xml tags
    :param accent_char_strip: bool, set to True to replace accented characters with their non accented versions
    :param lower_case: bool, set to True to lower case the document.
    :param no_newlines: bool, set to True to remove all newlines characters and replace them with spaces
    :param special_char_strip: bool, set to True to remove all characters that are
                                     not letters, numbers, $, ., %, or spaces
    :param lemmatize_words: bool, set to True to map each word to its lemma
    :param remove_stop_words: bool, set to True to remove stop words
    :param strip_extra_spaces: bool, set to True to replace multiple spaces with one
    :param remove_large_words: int, set to the integer cutoff where words larger than :param remove_large_words:
                               are removed from the text. Set to False if there is no cutoff

    ---> string, normalized document
    '''
    if debug:
        print('raw length of doc: {}'.format(len(doc)))

    # stripping tags
    if tags_strip:
        doc = strip_tags(doc)
    if debug:
        print('stripped tag length: {}'.format(len(doc)))

    # stripping accented characters
    if accent_char_strip:
        doc = strip_accented_chars(doc)
    if debug:
        print('altered accents length: {}'.format(len(doc)))

    # lower casing the document
    if lower_case:
        doc = doc.lower()
    if debug:
        print('lower casing length: {}'.format(len(doc)))

    # removing new lines and carriage returns and replacing them with spaces
    if no_newlines:
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
    if debug:
        print('removing newlines length: {}'.format(len(doc)))

    # removing special characters
    if special_char_strip:
        doc = strip_special_chars(doc)
    if debug:
        print('strip special char length: {}'.format(len(doc)))

    # lemmatizing the words
    if lemmatize_words:
        doc = lemmatize(doc)
    if debug:
        print('lemmatized doc length: {}'.format(len(doc)))

    # stripping stop words
    if remove_stop_words:
        doc = strip_stop_words(doc)
    if debug:
        print('removed stop word length: {}'.format(len(doc)))

    # adding spaces in between numbers and remaining special characters
    if space_nums:
        doc = re.sub(r'([\d$%.])', r' \1 ', doc)
    if debug:
        print('spaced num char length: {}'.format(len(doc)))

    # Removing large words
    if remove_large_words:
        doc = strip_large_words(doc, cut_off=remove_large_words)
    if debug:
        print('large words removed char length: {}'.format(len(doc)))

    # removing extra whitespace
    if strip_extra_spaces:
        doc = re.sub(' +', ' ', doc)
    if debug:
        print('removed extra space length: {}'.format(len(doc)))

    return doc
