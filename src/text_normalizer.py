import re
import subprocess
import unicodedata
from typing import List, Optional

import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer

from src.contractions import CONTRACTION_MAP

nltk.download("stopwords")
nltk.download("punkt")
subprocess.run(["spacy", "download", "en_core_web_sm"])
tokenizer = ToktokTokenizer()
nlp = spacy.load("en_core_web_sm")
stopword_list = nltk.corpus.stopwords.words("english")


def remove_html_tags(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text


def stem_text(text: str) -> str:
    stemmer = nltk.porter.PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_tokens)


def remove_accented_chars(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_special_chars(text: str, remove_digits: Optional[bool] = False) -> str:
    if remove_digits:
        pattern = r'[^a-zA-Z\s]'
    else:
        pattern = r'[^a-zA-Z0-9\s]'

    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def remove_stopwords(text: str, is_lower_case: Optional[bool] = False,
                     stopwords: Optional[List[str]] = stopword_list) -> str:
    if stopwords is None:
        stopwords = [] 

    if is_lower_case:
        text = text.lower()

    text = nltk.tokenize.word_tokenize(text)
    text = [w for w in text if w.lower() not in stopwords]
    str = " ".join(text)
    return str


def remove_extra_new_lines(text: str) -> str:
    return re.sub(r'[\r|\n|\r\n]+', ' ', text)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP) -> str:
    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = (
            contraction_mapping.get(match)
            if contraction_mapping.get(match)
            else contraction_mapping.get(match.lower())
        )
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    text = re.sub("'", "", expanded_text)

    return text


def normalize_corpus(
        corpus: List[str],
        html_stripping: Optional[bool] = True,
        contraction_expansion: Optional[bool] = True,
        accented_char_removal: Optional[bool] = True,
        text_lower_case: Optional[bool] = True,
        text_stemming: Optional[bool] = False,
        text_lemmatization: Optional[bool] = False,
        special_char_removal: Optional[bool] = True,
        remove_digits: Optional[bool] = True,
        stopword_removal: Optional[bool] = True,
        stopwords: Optional[List[str]] = stopword_list,
) -> List[str]:
    normalized_corpus = []

    for doc in corpus:
        if html_stripping:
            doc = remove_html_tags(doc)
        doc = remove_extra_new_lines(doc)
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        if contraction_expansion:
            doc = expand_contractions(doc)
        if text_lemmatization:
            doc = lemmatize_text(doc)
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
        if special_char_removal:
            doc = remove_special_chars(doc, remove_digits=remove_digits)
        doc = remove_extra_whitespace(doc)
        if text_lower_case:
            doc = doc.lower()
        if stopword_removal:
            doc = remove_stopwords(
                doc, is_lower_case=text_lower_case, stopwords=stopwords
            )
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()

        normalized_corpus.append(doc)

    return normalized_corpus
