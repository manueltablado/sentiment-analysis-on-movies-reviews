from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    num_docs = len(corpus)
    corpus_vectors = np.zeros((num_docs, num_features))

    for idx, doc_tokens in enumerate(corpus):
        doc_vector = np.zeros(num_features)
        num_tokens = 0

        for token in doc_tokens:
            if token in model.wv:
                doc_vector += model.wv[token]
                num_tokens += 1

        if num_tokens > 0:
            doc_vector /= num_tokens

        corpus_vectors[idx] = doc_vector

    return corpus_vectors
