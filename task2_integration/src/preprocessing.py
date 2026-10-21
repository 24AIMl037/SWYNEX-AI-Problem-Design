"""
EduIntent AI - Text Preprocessing Module
SWYNEX Technologies AI Internship - Task 2: Model Integration
Author: SWYNEX AI Intern
"""

import re
import string

# Domain-specific academic technical tokens to preserve
PRESERVE_TERMS = {
    "c++", "c#", ".net", "ci/cd", "i/o", "tcp/ip", "k-means", "k-nn", 
    "p-value", "n-gram", "tf-idf", "t-sne", "relu", "cnn", "rnn", "lstm",
    "dbms", "sql", "nosql", "acid", "b-tree", "ui/ux", "df", "np", "plt",
    "sns", "bert", "gpt", "api", "rest", "cpu", "ram", "ip", "udp", "http"
}

# Lightweight stopword list tailored for academic queries
# Retains query intent while filtering structural filler
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "should", "shouldn't",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won't",
    "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves",
    "please", "tell", "explain", "give", "help", "want", "know"
}


def clean_text(text: str) -> str:
    """
    Cleans raw student question text for model consumption:
    - Lowercases text
    - Strips URLs and HTML
    - Preserves programming terms and alphanumeric tokens
    - Normalizes multiple spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove HTML tags if any
    text = re.sub(r"<.*?>", " ", text)

    # Replace punctuation characters with spaces, except specific preserved hyphens/dots/plus/sharp
    text = re.sub(r"[^a-zA-Z0-9\s\-_+#]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_keywords(text: str, vectorizer=None, top_n: int = 5) -> list:
    """
    Extracts salient student-friendly keywords from the query.
    If a fitted TF-IDF vectorizer is provided, ranks words by their IDF relevance.
    Otherwise, extracts non-stopword alphanumeric tokens.
    """
    cleaned = clean_text(text)
    tokens = [tok for tok in cleaned.split() if tok not in STOPWORDS and len(tok) > 2]

    if not tokens:
        tokens = [tok for tok in cleaned.split() if len(tok) > 1]
    if not tokens:
        return ["query", "question"]

    if vectorizer is not None and hasattr(vectorizer, "vocabulary_"):
        vocab = vectorizer.vocabulary_
        idf_weights = getattr(vectorizer, "idf_", None)

        scored_tokens = []
        for tok in set(tokens):
            if tok in vocab:
                score = idf_weights[vocab[tok]] if idf_weights is not None else 1.0
                scored_tokens.append((tok, score))
            else:
                scored_tokens.append((tok, 0.5))

        # Sort descending by score
        scored_tokens.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored_tokens[:top_n]]

    # Fallback to unique tokens
    seen = set()
    unique_tokens = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            unique_tokens.append(tok)
    return unique_tokens[:top_n]
