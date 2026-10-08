"""ML response selector: scores candidate answers from multiple providers and picks the best.

Features: TF-IDF relevance to the question, length fit, structure, refusal/hedging signals,
repetition and latency. A logistic-regression model is trained on a small bundled preference
dataset at startup (see training_pairs below) so weights are learned, not hand-set.
"""
import math
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

REFUSAL = re.compile(r"\b(i can(no|')t|i am unable|i'm unable|as an ai|i do not have access|sorry, but)\b", re.I)
FEATURES = ["relevance", "length_fit", "structure", "refusal", "repetition", "latency"]


def featurize(question, answer, latency_ms=0):
    if not answer.strip():
        return np.zeros(len(FEATURES))
    try:
        vec = TfidfVectorizer(stop_words="english").fit([question, answer])
        rel = float(cosine_similarity(vec.transform([question]), vec.transform([answer]))[0, 0])
    except ValueError:
        rel = 0.0
    n = len(answer.split())
    length_fit = math.exp(-((math.log(n + 1) - math.log(120)) ** 2) / 2)  # peaks near ~120 words
    structure = min(1.0, (answer.count("\n- ") + answer.count("\n1.") + answer.count("**") / 2 + answer.count("```")) / 4)
    refusal = 1.0 if REFUSAL.search(answer) else 0.0
    words = answer.lower().split()
    repetition = 1 - len(set(words)) / max(len(words), 1)
    latency = min(latency_ms / 20000, 1.0)
    return np.array([rel, length_fit, structure, refusal, repetition, latency])


def _training_pairs():
    """(question, better, worse) preference triples. Small but enough to orient the weights."""
    return [
        ("How do I reverse a list in Python?",
         "Use slicing: `items[::-1]` returns a new reversed list, or `items.reverse()` reverses in place.\n- slicing keeps the original\n- reverse() is O(n) in place",
         "I'm unable to help with that."),
        ("Explain JWT authentication",
         "A JWT is a signed token with header, payload and signature. The server verifies the signature and expiry on each request.\n1. Login issues token\n2. Client sends Bearer token\n3. Server validates",
         "Authentication is important. Authentication is important. Authentication is important for apps."),
        ("What is SQL injection and how to prevent it?",
         "SQL injection happens when user input is concatenated into a query. Prevent it with parameterized queries, ORM bindings, least-privilege DB users and input validation.",
         "Sorry, but as an AI I cannot discuss that."),
        ("Summarize the benefits of rate limiting",
         "Rate limiting protects APIs from abuse and brute force, keeps costs predictable and ensures fair usage across clients.\n- token bucket\n- sliding window",
         "ok"),
        ("Compare Flask and Django",
         "Flask is a micro-framework: minimal core, pick your own ORM and auth. Django is batteries-included with ORM, admin and auth built in. Choose Flask for small APIs, Django for large CRUD apps.",
         "Both are frameworks. " * 40),
        ("What causes a CORS error?",
         "Browsers block cross-origin requests unless the server returns Access-Control-Allow-Origin for the requesting origin. Fix it on the server with the right CORS headers.",
         "Weather today is sunny with light winds in the afternoon."),
    ]


class ResponseRanker:
    def __init__(self):
        X, y = [], []
        for q, good, bad in _training_pairs():
            fg, fb = featurize(q, good, 1500), featurize(q, bad, 1500)
            X += [fg - fb, fb - fg]
            y += [1, 0]
        self.model = LogisticRegression(C=2.0).fit(np.array(X), np.array(y))
        self.weights = dict(zip(FEATURES, self.model.coef_[0].round(3).tolist()))

    def score(self, question, answer, latency_ms=0):
        return float(featurize(question, answer, latency_ms) @ self.model.coef_[0])

    def rank(self, question, results):
        """results: list of ProviderResult. Returns list of (result, score) best first."""
        scored = [(r, self.score(question, r.text, r.latency_ms) if r.ok and r.text.strip() else float("-inf"))
                  for r in results]
        return sorted(scored, key=lambda t: t[1], reverse=True)
