"""
Simple TF-IDF embedding utilities (no external ML dependencies).

Implements basic TF-IDF with cosine similarity for text comparison.
Uses simple whitespace tokenization and lowercase normalization.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization with lowercase normalization."""
    return text.lower().split()


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency for a tokenized document."""
    if not tokens:
        return {}
    tf: Dict[str, float] = {}
    for token in tokens:
        tf[token] = tf.get(token, 0.0) + 1.0
    # Normalize by document length
    doc_len = len(tokens)
    for token in tf:
        tf[token] /= doc_len
    return tf


def fit_tfidf(texts: List[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Compute vocabulary and IDF scores from a corpus of texts.

    Args:
        texts: List of documents to build the vocabulary from

    Returns:
        Tuple of (vocabulary dict mapping term to index, idf dict mapping term to idf score)
    """
    # Tokenize all documents
    tokenized_docs = [_tokenize(text) for text in texts]

    # Build vocabulary and document frequency
    vocab: Dict[str, int] = {}
    doc_freq: Dict[str, int] = {}
    vocab_idx = 0

    for tokens in tokenized_docs:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token not in vocab:
                vocab[token] = vocab_idx
                vocab_idx += 1
            doc_freq[token] = doc_freq.get(token, 0) + 1

    # Compute IDF: log(N / df) where N is total documents
    n_docs = len(texts)
    idf: Dict[str, float] = {}
    for token, df in doc_freq.items():
        idf[token] = math.log((n_docs + 1) / (df + 1)) + 1.0  # Smoothing

    return vocab, idf


def embed_text(text: str, vocab: Dict[str, int], idf: Dict[str, float]) -> List[float]:
    """
    Create TF-IDF embedding vector for a text.

    Args:
        text: Text to embed
        vocab: Vocabulary mapping term to index
        idf: IDF scores mapping term to idf value

    Returns:
        TF-IDF vector as list of floats
    """
    tokens = _tokenize(text)
    tf = _compute_tf(tokens)

    # Create vector of size vocab
    vector = [0.0] * len(vocab)

    for token, tf_score in tf.items():
        if token in vocab:
            idx = vocab[token]
            vector[idx] = tf_score * idf.get(token, 0.0)

    return vector


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1, 1], or 0.0 for zero vectors
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def compute_similarity_matrix(
    query_embedding: List[float],
    candidate_embeddings: List[List[float]],
) -> List[float]:
    """
    Compute cosine similarity between query and multiple candidates.

    Args:
        query_embedding: Query vector
        candidate_embeddings: List of candidate vectors

    Returns:
        List of similarity scores
    """
    return [cosine_similarity(query_embedding, candidate) for candidate in candidate_embeddings]
