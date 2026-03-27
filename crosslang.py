"""Cross-language entropy analysis: Linear B, Brown, and Europarl languages."""

import logging
import math
import sys

import nltk
import pandas as pd
import regex as reg
from nltk.corpus import brown, europarl_raw

from core import (
    MODEL_DIR,
    Q_GRAMS,
    build_kenlm_model,
    calculate_entropy_kenlm,
    calculate_redundancy,
    calculate_renyi_h2,
    calculate_unigram_entropy,
    cleanup_model,
    get_letter_filter,
    load_model,
)

CORPUS_PATH = MODEL_DIR.parent / "linearb_lexicon.csv"
LANGUAGES = [
    "english",
    "french",
    "german",
    "italian",
    "greek",
    "spanish",
    "dutch",
]
ALL_CORPORA = ["linear_b", "brown"] + LANGUAGES

# Ensure corpora are available
nltk.download("europarl_raw", quiet=True)
nltk.download("brown", quiet=True)


def load_linearb_corpus(csv_path=CORPUS_PATH):
    """Load Linear B words from CSV and format for KenLM."""
    df = pd.read_csv(csv_path)
    unique_words = df["word"].drop_duplicates()

    formatted = []
    all_chars = []
    for word in unique_words:
        chars = reg.findall(r"[\U00010000-\U000100FF]", word)
        if chars:
            formatted.append(" ".join(chars))
            all_chars.extend(chars)

    return "\n".join(formatted), len(unique_words), len(set(all_chars))


def load_brown_corpus():
    """Load Brown corpus with English letter filtering."""
    letter_filter = get_letter_filter("english")
    formatted = []
    all_chars = []

    for word in brown.words():
        cleaned = reg.sub(r"[^\p{L}]", "", word)
        if len(cleaned) < 3:
            continue
        cleaned = cleaned.lower()
        filtered = "".join(c for c in cleaned if letter_filter(c))
        if filtered:
            formatted.append(" ".join(filtered))
            all_chars.extend(filtered)

    text = "\n".join(formatted)
    vocab_count = len(set(w.lower() for w in brown.words()))
    return text, vocab_count, len(set(all_chars))


def load_europarl_corpus(language):
    """Load Europarl corpus for a language with language-specific letter filtering."""
    letter_filter = get_letter_filter(language)

    try:
        corpus = getattr(europarl_raw, language)
    except AttributeError:
        logging.error(f"Language '{language}' not found in europarl_raw.")
        return "", 0, 0

    formatted = []
    all_chars = []

    for sent in corpus.sents():
        for word in sent:
            cleaned = reg.sub(r"[^\p{L}]", "", word)
            if len(cleaned) < 3:
                continue
            cleaned = cleaned.lower()
            filtered = "".join(c for c in cleaned if letter_filter(c))
            if filtered:
                formatted.append(" ".join(filtered))
                all_chars.extend(filtered)

    text = "\n".join(formatted)
    vocab_count = len(
        set(
            w
            for s in corpus.sents()
            for w in s
            if len(reg.sub(r"[^\p{L}]", "", w)) >= 3
        )
    )
    return text, vocab_count, len(set(all_chars))


def analyze(corpus_name):
    """Run full entropy pipeline on one corpus. Returns redundancy or None."""
    logging.info(
        f"\nProcessing {corpus_name.replace('_', ' ').title()} Corpus"
    )

    if corpus_name == "linear_b":
        text, vocab_count, unique_letters = load_linearb_corpus()
        lang = "linear_b"
    elif corpus_name == "brown":
        text, vocab_count, unique_letters = load_brown_corpus()
        lang = "english"
    else:
        text, vocab_count, unique_letters = load_europarl_corpus(corpus_name)
        lang = corpus_name

    if len(text.split()) < Q_GRAMS:
        logging.error(f"Insufficient data in {corpus_name} corpus.")
        return None

    model_path = build_kenlm_model(text, corpus_name)
    if not model_path:
        logging.error(f"Failed to build model for {corpus_name}.")
        return None

    model = load_model(model_path)

    # Compute grapheme inventory from filtered text
    letter_filter = get_letter_filter(lang)
    text_letters = "".join(
        filter(letter_filter, text.replace(" ", "").replace("\n", ""))
    )
    if lang != "linear_b":
        text_letters = text_letters.lower()
    unique_set = set(text_letters)

    h_max = math.log2(len(unique_set)) if unique_set else 0
    h1 = calculate_unigram_entropy(text, lang)
    h2 = calculate_renyi_h2(text, lang)
    h3 = calculate_entropy_kenlm(model, text)
    redundancy = calculate_redundancy(h3, h_max) if h_max > 0 else 0

    logging.info(f"Vocab Count: {vocab_count}")
    logging.info(f"Grapheme Inventory: {len(unique_set)}")
    logging.info(f"Zero-order Entropy (H0): {h_max:.2f}")
    logging.info(f"First-order Entropy (H1): {h1:.2f}")
    logging.info(f"Second-order Entropy (H2): {h2:.2f}")
    logging.info(f"Third-order Entropy (H3) of {Q_GRAMS}-grams: {h3:.2f}")
    logging.info(f"Redundancy: {redundancy:.2f}%")

    cleanup_model(model_path)
    return redundancy


def run(corpora=None):
    """Analyze corpora and print summary redundancy table."""
    redundancies = {}
    for name in corpora or ALL_CORPORA:
        r = analyze(name)
        if r is not None:
            redundancies[name.replace("_", " ").title()] = r

    if redundancies:
        logging.info("\nRedundancy Rates Across Corpora:")
        for name, r in redundancies.items():
            logging.info(f"{name}: {r:.2f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(sys.argv[1:] or None)
