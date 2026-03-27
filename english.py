"""Entropy analysis for English NLTK corpora."""

import logging
import math
import sys

import nltk

from core import (
    Q_GRAMS,
    build_kenlm_model,
    calculate_entropy_kenlm,
    calculate_redundancy,
    calculate_renyi_h2,
    calculate_unigram_entropy,
    clean_and_format_words,
    cleanup_model,
    load_model,
)

ALL_CORPORA = [
    "brown",
    "reuters",
    "webtext",
    "inaugural",
    "nps_chat",
    "state_union",
    "gutenberg",
]


def load_corpus(corpus_name):
    """Load an NLTK corpus and return (kenlm_text, vocab_count, grapheme_inventory)."""
    words = getattr(nltk.corpus, corpus_name).words()
    cleaned = clean_and_format_words(words, "english")
    text = "\n".join(cleaned)
    all_chars = text.replace(" ", "").replace("\n", "")
    return text, len(set(cleaned)), len(set(all_chars))


def analyze(corpus_name):
    """Run full entropy pipeline on one corpus and log results."""
    text, vocab_count, grapheme_inventory = load_corpus(corpus_name)

    if len(text.split()) < Q_GRAMS:
        logging.error(f"Insufficient data in {corpus_name} corpus.")
        return

    model_path = build_kenlm_model(text, corpus_name)
    if not model_path:
        logging.error(f"Failed to build model for {corpus_name}.")
        return

    model = load_model(model_path)
    h_max = math.log2(grapheme_inventory) if grapheme_inventory > 0 else 0
    h1 = calculate_unigram_entropy(text, "english")
    h2 = calculate_renyi_h2(text, "english")
    h3 = calculate_entropy_kenlm(model, text)
    redundancy = calculate_redundancy(h3, h_max)

    logging.info(f"\nCorpus: {corpus_name}")
    logging.info(f"Token Count: {len(text.split())}")
    logging.info(f"Vocab Count: {vocab_count}")
    logging.info(f"Grapheme Inventory: {grapheme_inventory}")
    logging.info(f"Zero-order Entropy (H0): {h_max:.2f}")
    logging.info(f"First-order Entropy (H1): {h1:.2f}")
    logging.info(f"Second-order Entropy (H2): {h2:.2f}")
    logging.info(f"Third-order Entropy (H3) of {Q_GRAMS}-grams: {h3:.2f}")
    logging.info(f"Redundancy: {redundancy:.2f}%")

    cleanup_model(model_path)


def run(corpora=None):
    """Analyze one or more English corpora."""
    for name in corpora or ALL_CORPORA:
        analyze(name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(sys.argv[1:] or None)
