"""Sparse Vector Retrieval Module for Context Notes using TF-IDF.

Provides deterministic, offline lexical-semantic retrieval over the context note
corpus using term frequency-inverse document frequency (TF-IDF) and cosine similarity.
"""

from __future__ import annotations

from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.justification import ContextNote


def build_anomaly_query(route: str, week_of_str: str) -> str:
    """Construct search query for a flagged route-week anomaly.

    Combines corridor identity, week date, and freight disruption keywords.
    """
    origin, destination = route.split("-") if "-" in route else (route, "")
    return f"{route} {origin} {destination} {week_of_str} shipping freight cost rise disruption surcharge detour"


def retrieve_relevant_notes(
    route: str,
    week_of_str: str,
    notes: List[ContextNote],
    top_k: int = 3,
) -> List[Tuple[ContextNote, float]]:
    """Retrieve top-k relevant notes using TF-IDF sparse vector search.

    Args:
        route: Target corridor name (e.g. 'Chennai-Bangalore').
        week_of_str: Anomaly week Monday date (e.g. '2025-02-24').
        notes: Complete list of available ContextNote objects.
        top_k: Number of candidates to retrieve.

    Returns:
        List of (ContextNote, similarity_score) ranked by relevance.
    """
    scoped_notes = [
        n for n in notes if n.applies_to == "All Routes" or n.applies_to == route
    ]
    if not scoped_notes:
        return []

    corpus = [
        f"{n.note_id} {n.applies_to} {n.date_str} {n.note_text}" for n in scoped_notes
    ]
    query_text = build_anomaly_query(route, week_of_str)

    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus + [query_text])

    note_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1:]

    similarities = cosine_similarity(query_vector, note_vectors)[0]

    ranked = sorted(
        zip(scoped_notes, similarities),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:top_k]
