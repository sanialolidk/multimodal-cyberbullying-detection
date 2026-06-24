#!/usr/bin/env python3
"""
Multimodal Cyberbullying Detection: NLP + Interaction Network Features.

This script is intentionally self-contained. It builds TF-IDF features, simple
lexicon-based word embedding features, graph interaction features, and trains a
binary logistic regression classifier with NumPy.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "for",
    "from", "have", "i", "in", "is", "it", "lets", "me", "of", "on", "or",
    "please", "should", "that", "the", "this", "to", "was", "were", "will",
    "with", "you", "your",
}

ABUSE_LEXICON = {
    "ashamed", "block", "clown", "creep", "delete", "disgusting", "dumb",
    "embarrassing", "everyone", "idiot", "laugh", "loser", "mocked", "nobody",
    "pathetic", "sick", "shut", "stupid", "terrible", "trash", "ugly",
    "useless", "worthless",
}

POSITIVE_LEXICON = {
    "appreciate", "birthday", "charts", "clean", "clear", "congratulations",
    "debug", "feedback", "good", "great", "happy", "helpful", "nice",
    "respectful", "thanks", "wonderful",
}


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in normalize_text(text).split()
        if token not in STOP_WORDS and len(token) > 1
    ]


@dataclass
class FeatureArtifacts:
    vocabulary: list[str]
    idf: np.ndarray
    feature_names: list[str]
    means: np.ndarray
    stds: np.ndarray
    network_graph: dict = field(default_factory=dict)


def fit_tfidf(tokenized_docs: list[list[str]], max_features: int = 60) -> tuple[np.ndarray, list[str], np.ndarray]:
    doc_freq: Counter = Counter()
    term_counts = []
    for tokens in tokenized_docs:
        counts = Counter(tokens)
        term_counts.append(counts)
        doc_freq.update(counts.keys())

    vocabulary = [term for term, _ in doc_freq.most_common(max_features)]
    vocab_index = {term: idx for idx, term in enumerate(vocabulary)}
    n_docs = len(tokenized_docs)
    idf = np.array([
        math.log((1 + n_docs) / (1 + doc_freq[term])) + 1
        for term in vocabulary
    ])

    matrix = np.zeros((n_docs, len(vocabulary)), dtype=float)
    for row, counts in enumerate(term_counts):
        total = sum(counts.values()) or 1
        for term, count in counts.items():
            col = vocab_index.get(term)
            if col is not None:
                matrix[row, col] = (count / total) * idf[col]
    return matrix, vocabulary, idf


def transform_tfidf(tokenized_docs: list[list[str]], vocabulary: list[str], idf: np.ndarray) -> np.ndarray:
    vocab_index = {term: idx for idx, term in enumerate(vocabulary)}
    matrix = np.zeros((len(tokenized_docs), len(vocabulary)), dtype=float)
    for row, tokens in enumerate(tokenized_docs):
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        for term, count in counts.items():
            col = vocab_index.get(term)
            if col is not None:
                matrix[row, col] = (count / total) * idf[col]
    return matrix


def embedding_features(tokenized_docs: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    rows = []
    for tokens in tokenized_docs:
        token_count = max(len(tokens), 1)
        abusive_hits = sum(token in ABUSE_LEXICON for token in tokens)
        positive_hits = sum(token in POSITIVE_LEXICON for token in tokens)
        avg_token_length = sum(len(token) for token in tokens) / token_count
        rows.append([
            abusive_hits / token_count,
            positive_hits / token_count,
            avg_token_length,
            token_count,
        ])
    names = [
        "embed_abuse_density",
        "embed_positive_density",
        "embed_avg_token_length",
        "embed_token_count",
    ]
    return np.array(rows, dtype=float), names


def network_features_fit(df: pd.DataFrame) -> dict:
    """Build interaction graph statistics from training data only."""
    outgoing = df.groupby("user_id").size().to_dict()
    incoming = df.groupby("target_user_id").size().to_dict()
    pair_counts = df.groupby(["user_id", "target_user_id"]).size().to_dict()

    edges = pd.concat([
        df[["user_id", "target_user_id"]].rename(columns={"user_id": "a", "target_user_id": "b"}),
        df[["target_user_id", "user_id"]].rename(columns={"target_user_id": "a", "user_id": "b"}),
    ])
    neighbor_counts = edges.groupby("a")["b"].nunique().to_dict()

    return {
        "outgoing": outgoing,
        "incoming": incoming,
        "pair_counts": pair_counts,
        "neighbors": neighbor_counts,
    }


def network_features_transform(df: pd.DataFrame, graph: dict) -> tuple[np.ndarray, list[str]]:
    """Apply pre-fitted graph statistics to a DataFrame (unseen users get 0)."""
    outgoing = graph["outgoing"]
    incoming = graph["incoming"]
    pair_counts = graph["pair_counts"]
    neighbors = graph["neighbors"]

    rows = [
        [
            outgoing.get(row.user_id, 0),
            incoming.get(row.user_id, 0),
            incoming.get(row.target_user_id, 0),
            pair_counts.get((row.user_id, row.target_user_id), 0),
            neighbors.get(row.user_id, 0),
            neighbors.get(row.target_user_id, 0),
        ]
        for row in df.itertuples(index=False)
    ]
    names = [
        "net_author_out_degree",
        "net_author_in_degree",
        "net_target_in_degree",
        "net_pair_frequency",
        "net_author_neighbor_count",
        "net_target_neighbor_count",
    ]
    return np.array(rows, dtype=float), names


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1
    return (x - means) / stds, means, stds


def standardize_transform(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (x - means) / stds


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.25,
    epochs: int = 1200,
    l2: float = 0.01,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    n = len(y)

    for _ in range(epochs):
        probabilities = sigmoid(x @ weights + bias)
        error = probabilities - y
        weights -= learning_rate * ((x.T @ error) / n + l2 * weights)
        bias -= learning_rate * error.mean()
    return weights, bias


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_pred = (probabilities >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / len(y_true)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def stratified_train_test_split(
    y: np.ndarray,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for label in sorted(set(y.tolist())):
        label_indices = rng.permutation(np.where(y == label)[0])
        n_test = max(1, round(len(label_indices) * test_fraction))
        test_parts.append(label_indices[:n_test])
        train_parts.append(label_indices[n_test:])
    return np.concatenate(train_parts), np.concatenate(test_parts)


def make_features_fit(df: pd.DataFrame) -> tuple[np.ndarray, FeatureArtifacts]:
    """Fit all feature extractors on training data and return the feature matrix + artifacts."""
    tokens = [tokenize(text) for text in df["text"]]
    tfidf, vocabulary, idf = fit_tfidf(tokens)
    embeddings, embedding_names = embedding_features(tokens)
    graph = network_features_fit(df)
    net, graph_names = network_features_transform(df, graph)

    raw = np.hstack([tfidf, embeddings, net])
    scaled, means, stds = standardize_fit(raw)
    names = [f"tfidf_{term}" for term in vocabulary] + embedding_names + graph_names
    return scaled, FeatureArtifacts(vocabulary, idf, names, means, stds, graph)


def make_features_transform(df: pd.DataFrame, artifacts: FeatureArtifacts) -> np.ndarray:
    """Transform new data using artifacts fit on training data (no leakage)."""
    tokens = [tokenize(text) for text in df["text"]]
    tfidf = transform_tfidf(tokens, artifacts.vocabulary, artifacts.idf)
    embeddings, _ = embedding_features(tokens)
    net, _ = network_features_transform(df, artifacts.network_graph)

    raw = np.hstack([tfidf, embeddings, net])
    return standardize_transform(raw, artifacts.means, artifacts.stds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multimodal cyberbullying detector.")
    parser.add_argument("--data", default="sample_cyberbullying_data.csv",
                        help="CSV with post_id,user_id,target_user_id,text,label")
    parser.add_argument("--top-n", type=int, default=12,
                        help="Number of strongest model features to print")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    required = {"post_id", "user_id", "target_user_id", "text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["text"].isna().any():
        print(f"Warning: {df['text'].isna().sum()} row(s) have missing text — filling with empty string.")
        df["text"] = df["text"].fillna("")

    unique_labels = sorted(df["label"].unique())
    if set(unique_labels) - {0, 1}:
        raise ValueError(f"Labels must be binary (0/1). Found: {unique_labels}")

    y = df["label"].to_numpy(dtype=int)

    # Split BEFORE fitting any features to prevent data leakage
    train_idx, test_idx = stratified_train_test_split(y)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Fit on train only, transform test using train statistics
    x_train, artifacts = make_features_fit(df_train)
    x_test = make_features_transform(df_test, artifacts)

    weights, bias = train_logistic_regression(x_train, y_train)
    train_probs = sigmoid(x_train @ weights + bias)
    test_probs = sigmoid(x_test @ weights + bias)

    train_m = metrics(y_train, train_probs)
    test_m = metrics(y_test, test_probs)

    print("Multimodal Cyberbullying Detection")
    print("=" * 39)
    print(
        f"Rows: {len(df)} | Train: {len(df_train)} | Test: {len(df_test)} | "
        f"TF-IDF vocab: {len(artifacts.vocabulary)} | Features: {len(artifacts.feature_names)}"
    )
    print("\nTrain metrics")
    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {key:9s}: {train_m[key]:.3f}")
    print("\nTest metrics")
    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {key:9s}: {test_m[key]:.3f}")
    print(f"  confusion : TP={test_m['tp']} TN={test_m['tn']} FP={test_m['fp']} FN={test_m['fn']}")

    print(f"\nStrongest positive cyberbullying signals (top {args.top_n})")
    ranked = sorted(zip(artifacts.feature_names, weights), key=lambda x: x[1], reverse=True)
    for name, weight in ranked[: args.top_n]:
        print(f"  {name:28s} {weight:+.3f}")

    print("\nExample predictions")
    for i, row in enumerate(df_test.itertuples(index=False)):
        print(
            f"  {row.post_id}: true={y_test[i]} pred={(test_probs[i] >= 0.5):.0f} "
            f"prob={test_probs[i]:.3f} | {row.text}"
        )


if __name__ == "__main__":
    main()
