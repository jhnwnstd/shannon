"""Shared utilities for Shannon entropy analysis."""

import logging
import math
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional

import kenlm
import numpy as np
import regex

Q_GRAMS = 8
MODEL_DIR = Path(__file__).parent / "entropy_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# All supported language letter sets (merged from all scripts)
_LETTER_SETS = {
    "english": set("abcdefghijklmnopqrstuvwxyz"),
    "french": set("abcdefghijklmnopqrstuvwxyzàâçéèêëîïôûùüÿñæœ"),
    "german": set("abcdefghijklmnopqrstuvwxyzäöüß"),
    "italian": set("abcdefghijklmnopqrstuvwxyzàèéìíîòóùú"),
    "spanish": set("abcdefghijklmnopqrstuvwxyzñáéíóúü"),
    "dutch": set("abcdefghijklmnopqrstuvwxyzëï"),
    "greek": set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"),
    "danish": set(
        "abcdefghijklmnopqrstuvwxyzæøåABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ"
    ),
    "portuguese": set(
        "abcdefghijklmnopqrstuvwxyzçáàãâéêíóôõúABCDEFGHIJKLMNOPQRSTUVWXYZÇÁÀÃÂÉÊÍÓÔÕÚ"
    ),
    "romanian": set(
        "abcdefghijklmnopqrstuvwxyzăâîșțABCDEFGHIJKLMNOPQRSTUVWXYZĂÂÎȘȚ"
    ),
    "slovak": set(
        "abcdefghijklmnopqrstuvwxyzáčďéíľĺňóôŕšťúýžABCDEFGHIJKLMNOPQRSTUVWXYZÁČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ"
    ),
    "slovene": set(
        "abcdefghijklmnopqrstuvwxyzčšžABCDEFGHIJKLMNOPQRSTUVWXYZČŠŽ"
    ),
    "swedish": set(
        "abcdefghijklmnopqrstuvwxyzåäöABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ"
    ),
    "linear_b": set(chr(i) for i in range(0x10000, 0x100FF + 1)),
}


def get_letter_filter(language_name: str) -> Callable[[str], bool]:
    """Return a function that tests whether a character belongs to the language's alphabet."""
    letters = _LETTER_SETS.get(language_name.lower())
    if letters is None:
        return lambda char: regex.match(r"\p{L}", char) is not None
    return lambda char: char in letters


def clean_and_format_words(words: List[str], language_name: str) -> List[str]:
    """Clean words, apply language letter filter, space-separate characters for KenLM."""
    cleaned = []
    letter_filter = get_letter_filter(language_name)
    is_linear_b = language_name.lower() == "linear_b"
    for word in words:
        cleaned_word = regex.sub(r"[^\p{L}]", "", word)
        if len(cleaned_word) < 3:
            continue
        if not is_linear_b:
            cleaned_word = cleaned_word.lower()
        filtered = "".join(c for c in cleaned_word if letter_filter(c))
        if filtered:
            cleaned.append(" ".join(filtered))
    return cleaned


def build_kenlm_model(text: str, corpus_name: str) -> Optional[Path]:
    """Build a KenLM n-gram model from text. Returns path to binary .klm file."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        text_path = f.name

    arpa = MODEL_DIR / f"{corpus_name}_{Q_GRAMS}gram.arpa"
    binary = MODEL_DIR / f"{corpus_name}_{Q_GRAMS}gram.klm"

    ok = _run_cmd(
        f'lmplz -o {Q_GRAMS} --text "{text_path}" --arpa "{arpa}" --discount_fallback',
        "Failed to generate ARPA model",
    ) and _run_cmd(
        f'build_binary "{arpa}" "{binary}"',
        "Failed to convert ARPA to binary",
    )

    Path(text_path).unlink(missing_ok=True)
    return binary if ok else None


def load_model(model_path: Path) -> kenlm.Model:
    """Load a KenLM binary model."""
    return kenlm.Model(str(model_path))


def cleanup_model(model_path: Optional[Path]) -> None:
    """Delete KenLM model files (binary + ARPA)."""
    if not model_path:
        return
    try:
        Path(model_path).unlink(missing_ok=True)
        Path(model_path).with_suffix(".arpa").unlink(missing_ok=True)
    except Exception as e:
        logging.error(f"Failed to delete model files: {e}")


def calculate_entropy_kenlm(model: kenlm.Model, text: str) -> float:
    """Calculate H3 entropy using a KenLM model. Text should be space-separated tokens."""
    log_prob_bits = model.score(text, bos=False, eos=False) / math.log(2)
    num_ngrams = max(len(text.split()) - Q_GRAMS + 1, 1)
    return -log_prob_bits / num_ngrams


def calculate_unigram_entropy(text: str, language_name: str) -> float:
    """Calculate H1 (first-order / unigram entropy) in bits."""
    letter_filter = get_letter_filter(language_name)
    filtered = "".join(c for c in text if letter_filter(c))
    if language_name.lower() != "linear_b":
        filtered = filtered.lower()
    freq = Counter(filtered)
    total = sum(freq.values())
    if total == 0:
        return 0.0
    probs = np.array(list(freq.values())) / total
    return float(-np.sum(probs * np.log2(probs)))


def calculate_renyi_h2(text: str, language_name: str) -> float:
    """Calculate H2 (Renyi entropy of order 2) in bits."""
    letter_filter = get_letter_filter(language_name)
    filtered = "".join(c for c in text if letter_filter(c))
    if language_name.lower() != "linear_b":
        filtered = filtered.lower()
    freq = Counter(filtered)
    total = len(filtered)
    if total == 0:
        return 0.0
    probs = np.array(list(freq.values())) / total
    return float(-np.log2(np.sum(probs**2)))


def calculate_redundancy(h: float, h_max: float) -> float:
    """Calculate redundancy percentage: (1 - H/H0) * 100."""
    return (1 - h / h_max) * 100 if h_max > 0 else 0.0


def _run_cmd(command: str, error_message: str) -> bool:
    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(
            f"{error_message}: {e.stderr.decode().strip()} (exit code {e.returncode})"
        )
        return False
