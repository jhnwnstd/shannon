"""Comprehensive entropy analysis with per-file statistics, digrams, trigrams, and transitions."""

import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
from tqdm import tqdm

from core import (
    build_kenlm_model,
    calculate_entropy_kenlm,
    calculate_redundancy,
    clean_and_format_words,
    cleanup_model,
    load_model,
)

LANGUAGE_CODE_MAP = {
    "bg": "bulgarian",
    "cs": "czech",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "hu": "hungarian",
    "it": "italian",
    "lt": "lithuanian",
    "lv": "latvian",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "sk": "slovak",
    "sl": "slovene",
    "sv": "swedish",
}

STANDARD_CORPORA = [
    "brown",
    "reuters",
    "webtext",
    "inaugural",
    "nps_chat",
    "state_union",
    "gutenberg",
]

ALL_ENTRIES = [
    ("brown", None),
    ("reuters", None),
    ("webtext", None),
    ("inaugural", None),
    ("nps_chat", None),
    ("state_union", None),
    ("gutenberg", None),
    ("europarl_raw", "en"),
    ("europarl_raw", "de"),
    ("europarl_raw", "fr"),
    ("europarl_raw", "es"),
    ("europarl_raw", "it"),
    ("europarl_raw", "nl"),
    ("europarl_raw", "pt"),
    ("europarl_raw", "sv"),
    ("europarl_raw", "da"),
    ("europarl_raw", "fi"),
    ("europarl_raw", "el"),
]


@dataclass
class EntropyResults:
    """Per-file entropy calculation results."""

    h0: float
    h1: float
    h2: float
    h3: float
    alphabet_size: int
    unique_digrams: int
    unique_trigrams: int
    total_chars: int
    char_distribution: Dict[str, float]
    digram_distribution: Dict[str, float]
    trigram_distribution: Dict[str, float]
    transitions: Dict[str, Dict[str, float]]
    markov_efficiency: float
    compression_ratio: float
    predictability: float
    branching_factor: float
    char_freq: Counter
    digram_freq: Counter
    trigram_freq: Counter


@dataclass
class CorpusStatistics:
    """Aggregated corpus-level statistics."""

    corpus_name: str
    files_analyzed: int
    total_chars: int
    mean_entropy: Dict[str, float]
    std_entropy: Dict[str, float]
    reductions: Dict[str, float]
    patterns: Dict[str, Dict[str, float]]
    efficiency_metrics: Dict[str, float]


class ShannonAnalyzer:
    def __init__(self, ngram_order: int = 8):
        self.ngram_order = ngram_order
        self.logger = logging.getLogger(__name__)
        self._download_corpora()

    def _download_corpora(self) -> None:
        required = STANDARD_CORPORA + ["europarl_raw"]
        for corpus in required:
            try:
                nltk.data.find(f"corpora/{corpus}")
            except LookupError:
                self.logger.info(f"Downloading {corpus} corpus...")
                nltk.download(corpus)

    def preprocess_text(self, text: str, language_name: str) -> List[str]:
        return clean_and_format_words(text.split(), language_name)

    def calculate_ngram_stats(
        self, words: List[str], n: int
    ) -> Tuple[Counter, int]:
        ngram_counter: Counter[str] = Counter()
        for word in words:
            letters = word.split()
            for i in range(len(letters) - n + 1):
                ngram_counter["".join(letters[i : i + n])] += 1
        return ngram_counter, sum(ngram_counter.values())

    def calculate_entropy(
        self,
        freq: Counter,
        total: int,
        prev_freq: Optional[Counter] = None,
        prev_total: Optional[int] = None,
    ) -> float:
        """Entropy from n-gram frequencies. With prev_freq/prev_total, computes conditional entropy."""
        if prev_freq is None:
            probs = [count / total for count in freq.values()]
            return -sum(p * np.log2(p) for p in probs if p > 0)

        entropy = 0.0
        for seq, count in freq.items():
            prefix = seq[:-1]
            p_seq = count / total
            p_prev = (
                prev_freq[prefix] / prev_total
                if prev_total and prefix in prev_freq
                else 0
            )
            if p_prev > 0 and p_seq > 0:
                entropy -= p_seq * np.log2(p_seq / p_prev)
        return entropy

    def analyze_text(
        self, formatted_words: List[str], language_name: str
    ) -> EntropyResults:
        """Analyze a single text file: entropy, digrams, trigrams, transitions."""
        if not formatted_words:
            raise ValueError("No valid words to analyze.")

        char_freq, total_chars = self.calculate_ngram_stats(formatted_words, 1)
        digram_freq, total_digrams = self.calculate_ngram_stats(
            formatted_words, 2
        )
        trigram_freq, total_trigrams = self.calculate_ngram_stats(
            formatted_words, 3
        )

        h0 = np.log2(len(char_freq)) if len(char_freq) > 0 else 0
        h1 = self.calculate_entropy(char_freq, total_chars)
        h2 = self.calculate_entropy(
            digram_freq, total_digrams, char_freq, total_chars
        )

        char_dist = {
            c: count / total_chars for c, count in char_freq.most_common(10)
        }

        transitions: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for digram, count in digram_freq.items():
            transitions[digram[0]][digram[1]] = (
                count / char_freq[digram[0]]
                if char_freq[digram[0]] > 0
                else 0.0
            )

        markov_efficiency = 100 * (h1 - h2) / h1 if h1 > 0 else 0.0
        branching_factor = float(
            np.mean([len(t) for t in transitions.values()])
            if transitions
            else 0.0
        )

        return EntropyResults(
            h0=h0,
            h1=h1,
            h2=h2,
            h3=0.0,
            alphabet_size=len(char_freq),
            unique_digrams=len(digram_freq),
            unique_trigrams=len(trigram_freq),
            total_chars=total_chars,
            char_distribution=char_dist,
            digram_distribution={
                k: c / total_digrams for k, c in digram_freq.most_common(10)
            },
            trigram_distribution={
                k: c / total_trigrams for k, c in trigram_freq.most_common(10)
            },
            transitions={k: dict(v) for k, v in transitions.items()},
            markov_efficiency=markov_efficiency,
            compression_ratio=0.0,
            predictability=0.0,
            branching_factor=branching_factor,
            char_freq=char_freq,
            digram_freq=digram_freq,
            trigram_freq=trigram_freq,
        )

    def analyze_corpus_with_kenlm(
        self,
        corpus_name: str,
        language_code: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> CorpusStatistics:
        """Full corpus analysis with KenLM H3 and aggregated statistics."""
        if corpus_name == "europarl_raw" and not language_code:
            raise ValueError("Language code required for europarl_raw.")

        if corpus_name == "europarl_raw":
            language_name = LANGUAGE_CODE_MAP.get(language_code or "")
            if not language_name:
                raise ValueError(f"Unsupported language code: {language_code}")
            display_name = (
                f"europarl_raw.{language_code} ({language_name.capitalize()})"
            )
        else:
            language_name = corpus_name
            display_name = corpus_name.capitalize()

        self.logger.info(f"Starting analysis of '{display_name}' corpus...")

        # Load file IDs
        corpus_reader = None
        corpus = None
        if corpus_name == "europarl_raw":
            from nltk.corpus import europarl_raw

            corpus_reader = getattr(europarl_raw, language_name)
            file_ids = (
                corpus_reader.fileids()[:max_files]
                if max_files
                else corpus_reader.fileids()
            )
        else:
            corpus = getattr(nltk.corpus, corpus_name)
            file_ids = (
                corpus.fileids()[:max_files] if max_files else corpus.fileids()
            )

        if not file_ids:
            self.logger.warning(f"No files found for '{display_name}'.")
            return CorpusStatistics(
                corpus_name=display_name,
                files_analyzed=0,
                total_chars=0,
                mean_entropy={},
                std_entropy={},
                reductions={},
                patterns={},
                efficiency_metrics={},
            )

        results = []
        total_chars = 0
        total_char_freq: Counter[str] = Counter()
        total_digram_freq: Counter[str] = Counter()
        aggregated_text = ""

        for file_id in tqdm(
            file_ids, desc=f"Analyzing '{display_name}' files"
        ):
            try:
                if corpus_name == "europarl_raw":
                    if corpus_reader is None:
                        raise ValueError("corpus_reader is None for europarl_raw")
                    text = corpus_reader.raw(file_id)
                else:
                    if corpus is None:
                        raise ValueError("corpus is None")
                    text = corpus.raw(file_id)

                formatted_words = self.preprocess_text(text, language_name)
                result = self.analyze_text(formatted_words, language_name)
                results.append(result)
                total_chars += result.total_chars
                total_char_freq.update(result.char_freq)
                total_digram_freq.update(result.digram_freq)
                aggregated_text += "\n".join(formatted_words) + "\n"
            except Exception as e:
                self.logger.warning(f"Error processing '{file_id}': {e}")

        if total_chars == 0:
            self.logger.warning(f"No characters found in '{display_name}'.")
            return CorpusStatistics(
                corpus_name=display_name,
                files_analyzed=len(results),
                total_chars=0,
                mean_entropy={},
                std_entropy={},
                reductions={},
                patterns={},
                efficiency_metrics={},
            )

        # Aggregated entropy
        h0 = float(np.log2(len(total_char_freq))) if total_char_freq else 0
        h1 = self.calculate_entropy(total_char_freq, total_chars)

        total_digrams = sum(total_digram_freq.values())
        h2 = self.calculate_entropy(
            total_digram_freq, total_digrams, total_char_freq, total_chars
        )

        markov_efficiency = 100 * (h1 - h2) / h1 if h1 > 0 else 0.0

        transitions = defaultdict(set)
        for digram in total_digram_freq:
            transitions[digram[0]].add(digram[1])
        branching_factor = float(
            np.mean([len(v) for v in transitions.values()])
            if transitions
            else 0.0
        )

        # Per-file mean/std
        mean_entropy: Dict[str, float] = {
            "h0": float(np.mean([r.h0 for r in results])),
            "h1": float(np.mean([r.h1 for r in results])),
            "h2": float(np.mean([r.h2 for r in results])),
        }
        std_entropy: Dict[str, float] = {
            "h0": float(np.std([r.h0 for r in results])),
            "h1": float(np.std([r.h1 for r in results])),
            "h2": float(np.std([r.h2 for r in results])),
        }

        reductions: Dict[str, float] = {
            "h0_to_h1": float(
                100 * (1 - mean_entropy["h1"] / mean_entropy["h0"])
                if mean_entropy["h0"] > 0
                else 0.0
            ),
            "h1_to_h2": float(
                100 * (1 - mean_entropy["h2"] / mean_entropy["h1"])
                if mean_entropy["h1"] > 0
                else 0.0
            ),
        }

        char_distribution = {
            c: count / total_chars
            for c, count in total_char_freq.most_common(10)
        }
        patterns = {
            "chars": dict(
                sorted(
                    char_distribution.items(), key=lambda x: x[1], reverse=True
                )[:5]
            )
        }

        # KenLM H3
        self.logger.info("Building KenLM model for H3 entropy calculation...")
        model_path = build_kenlm_model(aggregated_text, display_name)
        h3_kenlm = 0.0
        if model_path and model_path.exists():
            try:
                model = load_model(model_path)
                h3_kenlm = calculate_entropy_kenlm(model, aggregated_text)
                self.logger.info(f"KenLM H3 Entropy: {h3_kenlm:.2f} bits")
                self.logger.info(
                    f"Redundancy: {calculate_redundancy(h3_kenlm, h0):.2f}%"
                )
            except Exception as e:
                self.logger.error(f"KenLM H3 failed for '{display_name}': {e}")
        else:
            self.logger.error(
                f"KenLM model creation failed for '{display_name}'."
            )

        # Update stats with H3
        mean_entropy["h3"] = float(h3_kenlm)
        std_entropy["h3"] = 0.0

        reductions["h2_to_h3"] = float(
            100 * (1 - mean_entropy["h3"] / mean_entropy["h2"])
            if mean_entropy["h2"] > 0
            else 0.0
        )
        reductions["total"] = float(
            100 * (1 - mean_entropy["h3"] / mean_entropy["h0"])
            if mean_entropy["h0"] > 0
            else 0.0
        )

        compression_ratio = float(
            h3_kenlm / mean_entropy["h0"] if mean_entropy["h0"] > 0 else 0.0
        )
        predictability = float(
            100 * (1 - h3_kenlm / mean_entropy["h0"])
            if mean_entropy["h0"] > 0
            else 0.0
        )

        efficiency_metrics: Dict[str, float] = {
            "markov_efficiency": float(markov_efficiency),
            "compression_ratio": compression_ratio,
            "predictability": predictability,
            "branching_factor": float(branching_factor),
        }

        cleanup_model(model_path)

        return CorpusStatistics(
            corpus_name=display_name,
            files_analyzed=len(results),
            total_chars=total_chars,
            mean_entropy=mean_entropy,
            std_entropy=std_entropy,
            reductions=reductions,
            patterns=patterns,
            efficiency_metrics=efficiency_metrics,
        )


def print_analysis(stats: CorpusStatistics) -> None:
    """Pretty-print corpus analysis results."""
    print(f"\n{stats.corpus_name} Corpus Analysis")
    print("=" * 50)
    print(f"Files analyzed: {stats.files_analyzed}")
    print(f"Total characters: {stats.total_chars:,}")

    print("\nEntropy Measures (bits)")
    print("-" * 30)
    for order in ["h0", "h1", "h2", "h3"]:
        mean = stats.mean_entropy.get(order, 0.0)
        std = stats.std_entropy.get(order, 0.0)
        print(f"{order.upper()}: {mean:.2f} +/- {std:.2f}")

    print("\nInformation Reduction")
    print("-" * 30)
    for key, value in stats.reductions.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value:.1f}%")

    print("\nEfficiency Metrics")
    print("-" * 30)
    for key, value in stats.efficiency_metrics.items():
        label = key.replace("_", " ").capitalize()
        print(
            f"{label}: {value:.2f}"
            if "ratio" in key
            else f"{label}: {value:.1f}%"
        )

    print("\nMost Common Characters")
    print("-" * 30)
    for char, freq in stats.patterns.get("chars", {}).items():
        print(f"'{char}': {freq * 100:.1f}%")


def run(args=None):
    """Run analysis. Args are corpus names or language codes to filter by."""
    if args:
        requested = set(args)
        entries = [
            (c, lc)
            for c, lc in ALL_ENTRIES
            if c in requested or (lc and lc in requested)
        ]
    else:
        entries = ALL_ENTRIES

    analyzer = ShannonAnalyzer(ngram_order=8)
    for corpus, lang_code in entries:
        try:
            if corpus == "europarl_raw":
                stats = analyzer.analyze_corpus_with_kenlm(
                    corpus, language_code=lang_code
                )
            else:
                stats = analyzer.analyze_corpus_with_kenlm(corpus)
            print_analysis(stats)
        except Exception as e:
            logging.error(
                f"Failed to analyze '{corpus}' (lang={lang_code}): {e}"
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    run(sys.argv[1:] or None)
