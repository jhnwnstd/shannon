import numpy as np
from collections import Counter, defaultdict
import regex  # Ensure using the third-party regex module
from typing import Dict, List, Optional, Tuple
import nltk
from nltk.corpus import *
import logging
from tqdm import tqdm
from dataclasses import dataclass
import sys

# ============================
# Helper Functions and Data Classes
# ============================

@dataclass
class EntropyResults:
    """Container for entropy calculation results"""
    h0: float  # Maximum possible entropy
    h1: float  # First-order entropy
    h2: float  # Second-order entropy
    h3: float  # Third-order entropy
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
    """Container for corpus-level statistics"""
    corpus_name: str
    files_analyzed: int
    total_chars: int
    mean_entropy: Dict[str, float]
    std_entropy: Dict[str, float]
    reductions: Dict[str, float]
    patterns: Dict[str, Dict[str, float]]
    efficiency_metrics: Dict[str, float]

def get_letter_filter(corpus_name: str) -> callable:
    """
    Returns a filtering function for letters based on the corpus name.
    """
    # Define allowed letters per language
    allowed_letters = {
        'english': set('abcdefghijklmnopqrstuvwxyz'),
        'french': set('abcdefghijklmnopqrstuvwxyzàâçéèêëîïôûùüÿñæœ'),
        'german': set('abcdefghijklmnopqrstuvwxyzäöüß'),
        'italian': set('abcdefghijklmnopqrstuvwxyzàèéìíîòóùú'),
        'spanish': set('abcdefghijklmnopqrstuvwxyzñáéíóúü'),
        'dutch': set('abcdefghijklmnopqrstuvwxyzëï'),
        'greek': set('αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'),
        'linear_b': set(chr(i) for i in range(0x10000, 0x100FF + 1)),
    }

    corpus_name_lower = corpus_name.lower()
    letters = allowed_letters.get(corpus_name_lower)

    if letters is None:
        # Default to all Unicode letters if the corpus is unrecognized
        return lambda char: regex.match(r'\p{L}', char) is not None
    else:
        # Return a filter function that checks if the character is in the allowed set
        return lambda char: char in letters

def clean_and_format_words(words: List[str], corpus_name: str) -> List[str]:
    """
    Clean and format words by removing non-letter characters, converting to lowercase, and separating letters with spaces.
    """
    cleaned_words = []
    letter_filter = get_letter_filter(corpus_name)
    for word in words:
        try:
            # Remove non-letter characters using regex
            cleaned_word = regex.sub(r'[^\p{L}]', '', word)
            if len(cleaned_word) >= 3:
                # Convert to lowercase if not Linear B
                if corpus_name.lower() != 'linear_b':
                    cleaned_word = cleaned_word.lower()
                # Filter letters
                filtered_letters = ''.join([char for char in cleaned_word if letter_filter(char)])
                if filtered_letters:
                    # Treat each letter as a separate token by joining with spaces
                    formatted_word = ' '.join(filtered_letters)
                    cleaned_words.append(formatted_word)
        except regex.error as regex_err:
            print(f"Regex error while processing word '{word}': {regex_err}", file=sys.stderr)
    return cleaned_words

# ============================
# ShannonAnalyzer Class
# ============================

class ShannonAnalyzer:
    def __init__(self, ngram_order: int = 8):
        self.ngram_order = ngram_order
        self._setup_logging()
        self._download_corpora()
        
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _download_corpora(self) -> None:
        required = ['gutenberg', 'brown', 'reuters', 'europarl_raw']
        for corpus in required:
            try:
                nltk.data.find(f'corpora/{corpus}')
            except LookupError:
                self.logger.info(f"Downloading {corpus} corpus...")
                nltk.download(corpus)

    def preprocess_text(self, text: str, corpus_name: str) -> List[str]:
        """
        Preprocess text by cleaning, filtering, and formatting.
        """
        words = text.split()
        cleaned_formatted_words = clean_and_format_words(words, corpus_name)
        return cleaned_formatted_words

    def calculate_ngram_stats(self, words: List[str], n: int) -> Tuple[Counter, int]:
        """
        Calculate n-gram frequencies and total number of n-grams.
        """
        ngram_counter = Counter()
        for word in words:
            letters = word.split()
            if len(letters) < n:
                continue
            for i in range(len(letters) - n + 1):
                ngram = ''.join(letters[i:i+n])
                ngram_counter[ngram] += 1
        total_ngrams = sum(ngram_counter.values())
        return ngram_counter, total_ngrams

    def calculate_entropy(self, freq: Counter, total: int, 
                          prev_freq: Optional[Counter] = None,
                          prev_total: Optional[int] = None) -> float:
        """
        Calculate entropy based on n-gram frequencies.
        """
        if prev_freq is None:
            # Zero-order or first-order entropy
            probs = [count / total for count in freq.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            return entropy
        
        # Conditional entropy
        entropy = 0.0
        for seq, count in freq.items():
            prefix = seq[:-1]
            p_seq = count / total
            p_prev = prev_freq[prefix] / prev_total if prefix in prev_freq else 0
            if p_prev > 0 and p_seq > 0:
                entropy -= p_seq * np.log2(p_seq / p_prev)
        return entropy

    def analyze_text(self, formatted_words: List[str], corpus_name: str) -> EntropyResults:
        """
        Analyze a single text to calculate entropy and related metrics.
        """
        if not formatted_words:
            raise ValueError("No valid words to analyze.")
        
        # Calculate n-gram statistics
        char_freq, total_chars = self.calculate_ngram_stats(formatted_words, 1)
        digram_freq, total_digrams = self.calculate_ngram_stats(formatted_words, 2)
        trigram_freq, total_trigrams = self.calculate_ngram_stats(formatted_words, 3)
        
        # Calculate entropies
        h0 = np.log2(len(char_freq)) if len(char_freq) > 0 else 0
        h1 = self.calculate_entropy(char_freq, total_chars)
        h2 = self.calculate_entropy(digram_freq, total_digrams, char_freq, total_chars)
        h3 = self.calculate_entropy(trigram_freq, total_trigrams, digram_freq, total_digrams)
        
        # Calculate distributions (Top 10 for simplicity)
        char_dist = {c: count / total_chars for c, count in char_freq.most_common(10)}
        
        # Calculate transition matrices for digrams
        transitions = defaultdict(lambda: defaultdict(float))
        for digram, count in digram_freq.items():
            first, second = digram[0], digram[1]
            transitions[first][second] = count / char_freq[first] if char_freq[first] > 0 else 0.0
        
        # Calculate advanced metrics
        markov_efficiency = 100 * (h1 - h2) / h1 if h1 > 0 else 0.0
        compression_ratio = h3 / h0 if h0 > 0 else 0.0
        predictability = 100 * (1 - h3 / h0) if h0 > 0 else 0.0
        branching_factor = np.mean([len(trans) for trans in transitions.values()]) if transitions else 0.0
        
        return EntropyResults(
            h0=h0,
            h1=h1,
            h2=h2,
            h3=h3,
            alphabet_size=len(char_freq),
            unique_digrams=len(digram_freq),
            unique_trigrams=len(trigram_freq),
            total_chars=total_chars,
            char_distribution=char_dist,
            digram_distribution={k: count / total_digrams for k, count in digram_freq.most_common(10)},
            trigram_distribution={k: count / total_trigrams for k, count in trigram_freq.most_common(10)},
            transitions={k: dict(v) for k, v in transitions.items()},
            markov_efficiency=markov_efficiency,
            compression_ratio=compression_ratio,
            predictability=predictability,
            branching_factor=branching_factor,
            char_freq=char_freq,          # Absolute counts
            digram_freq=digram_freq,      # Absolute counts
            trigram_freq=trigram_freq     # Absolute counts
        )

    def analyze_corpus(self, corpus_name: str, max_files: Optional[int] = None) -> CorpusStatistics:
        """
        Analyze an entire corpus to compute entropy and redundancy metrics.
        """
        self.logger.info(f"Starting analysis of '{corpus_name}' corpus...")
        
        if corpus_name.lower() == 'linear_b':
            # Handle Linear B separately if needed
            raise NotImplementedError("Linear B analysis should be handled in 'linearb_entro.py'.")
        
        corpus = getattr(nltk.corpus, corpus_name)
        file_ids = corpus.fileids()[:max_files] if max_files else corpus.fileids()
        
        results = []
        total_chars = 0
        total_char_freq = Counter()
        total_digram_freq = Counter()
        total_trigram_freq = Counter()
        
        for file_id in tqdm(file_ids, desc=f"Analyzing '{corpus_name}' files"):
            try:
                text = corpus.raw(file_id)
                formatted_words = self.preprocess_text(text, corpus_name)
                result = self.analyze_text(formatted_words, corpus_name)
                results.append(result)
                total_chars += result.total_chars
                
                # Aggregate absolute character counts
                total_char_freq.update(result.char_freq)
                
                # Aggregate absolute digram and trigram counts
                total_digram_freq.update(result.digram_freq)
                total_trigram_freq.update(result.trigram_freq)
                
            except Exception as e:
                self.logger.warning(f"Error processing '{file_id}': {str(e)}")
        
        if total_chars == 0:
            self.logger.warning(f"No characters found in corpus '{corpus_name}'.")
            return CorpusStatistics(
                corpus_name=corpus_name,
                files_analyzed=len(results),
                total_chars=total_chars,
                mean_entropy={},
                std_entropy={},
                reductions={},
                patterns={},
                efficiency_metrics={}
            )
        
        # Now, calculate aggregated character distribution (Top 10)
        char_distribution = {c: count / total_chars for c, count in total_char_freq.most_common(10)}
        
        # Calculate overall entropy measures based on aggregated counts
        aggregated_char_freq = total_char_freq
        h0 = np.log2(len(aggregated_char_freq)) if len(aggregated_char_freq) > 0 else 0
        h1 = self.calculate_entropy(aggregated_char_freq, total_chars)
        
        # Aggregating digram and trigram counts
        aggregated_digram_freq = total_digram_freq
        aggregated_trigram_freq = total_trigram_freq
        
        total_digrams = sum(aggregated_digram_freq.values())
        total_trigrams = sum(aggregated_trigram_freq.values())
        
        h2 = self.calculate_entropy(aggregated_digram_freq, total_digrams, aggregated_char_freq, total_chars)
        h3 = self.calculate_entropy(aggregated_trigram_freq, total_trigrams, aggregated_digram_freq, total_digrams)
        
        # Calculate advanced metrics based on aggregated entropies
        markov_efficiency = 100 * (h1 - h2) / h1 if h1 > 0 else 0.0
        compression_ratio = h3 / h0 if h0 > 0 else 0.0
        predictability = 100 * (1 - h3 / h0) if h0 > 0 else 0.0
        
        # Corrected branching_factor calculation
        # Build mapping from first character to set of unique second characters
        transitions = defaultdict(set)
        for digram in aggregated_digram_freq:
            first, second = digram[0], digram[1]
            transitions[first].add(second)
        
        branching_factor = np.mean([len(v) for v in transitions.values()]) if transitions else 0.0
        
        # Calculate mean and std for entropy measures across all files
        mean_entropy = {
            'h0': np.mean([r.h0 for r in results]),
            'h1': np.mean([r.h1 for r in results]),
            'h2': np.mean([r.h2 for r in results]),
            'h3': np.mean([r.h3 for r in results])
        }
        std_entropy = {
            'h0': np.std([r.h0 for r in results]),
            'h1': np.std([r.h1 for r in results]),
            'h2': np.std([r.h2 for r in results]),
            'h3': np.std([r.h3 for r in results])
        }
        
        # Calculate reductions
        reductions = {
            'h0_to_h1': 100 * (1 - mean_entropy['h1'] / mean_entropy['h0']) if mean_entropy['h0'] > 0 else 0.0,
            'h1_to_h2': 100 * (1 - mean_entropy['h2'] / mean_entropy['h1']) if mean_entropy['h1'] > 0 else 0.0,
            'h2_to_h3': 100 * (1 - mean_entropy['h3'] / mean_entropy['h2']) if mean_entropy['h2'] > 0 else 0.0,
            'total': 100 * (1 - mean_entropy['h3'] / mean_entropy['h0']) if mean_entropy['h0'] > 0 else 0.0
        }

        # Calculate efficiency metrics
        efficiency_metrics = {
            'markov_efficiency': markov_efficiency,
            'compression_ratio': compression_ratio,
            'predictability': predictability,
            'branching_factor': branching_factor
        }

        # Aggregate patterns (Top 5 characters)
        patterns = {
            'chars': dict(sorted(char_distribution.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:5])
        }

        return CorpusStatistics(
            corpus_name=corpus_name,
            files_analyzed=len(results),
            total_chars=total_chars,
            mean_entropy=mean_entropy,
            std_entropy=std_entropy,
            reductions=reductions,
            patterns=patterns,
            efficiency_metrics=efficiency_metrics
        )

def print_analysis(stats: CorpusStatistics) -> None:
    """
    Print the analysis results in a formatted manner.
    """
    print(f"\n{stats.corpus_name.capitalize()} Corpus Analysis")
    print("=" * 50)
    print(f"Files analyzed: {stats.files_analyzed}")
    print(f"Total characters: {stats.total_chars:,}")
    
    print("\nEntropy Measures (bits)")
    print("-" * 30)
    for order in ['h0', 'h1', 'h2', 'h3']:
        mean = stats.mean_entropy.get(order, 0.0)
        std = stats.std_entropy.get(order, 0.0)
        print(f"{order.upper()}: {mean:.2f} ± {std:.2f}")
    
    print("\nInformation Reduction")
    print("-" * 30)
    for reduction, value in stats.reductions.items():
        formatted = reduction.replace('_', ' ').capitalize()
        print(f"{formatted}: {value:.1f}%")
    
    print("\nEfficiency Metrics")
    print("-" * 30)
    for metric, value in stats.efficiency_metrics.items():
        formatted = metric.replace('_', ' ').capitalize()
        if 'ratio' in metric:
            print(f"{formatted}: {value:.2f}")
        else:
            print(f"{formatted}: {value:.1f}%")
    
    print("\nMost Common Characters")
    print("-" * 30)
    for char, freq in stats.patterns.get('chars', {}).items():
        print(f"'{char}': {freq*100:.1f}%")

# ============================
# Main Execution
# ============================

if __name__ == "__main__":
    # Debugging: Verify regex module is correctly imported
    try:
        import regex
        print(f"Using regex module: {regex.__name__}")
        print(f"Regex module version: {regex.__version__}")
    except ImportError as e:
        print(f"Regex module is not installed: {e}", file=sys.stderr)
        sys.exit(1)

    analyzer = ShannonAnalyzer(ngram_order=8)
    
    # Define the list of corpora to analyze
    # Exclude 'linear_b' as it should be handled separately in 'linearb_entro.py'
    CORPORA = ['gutenberg', 'brown', 'reuters']
    
    for corpus in CORPORA:
        try:
            stats = analyzer.analyze_corpus(corpus, max_files=None)  # Set max_files as needed
            print_analysis(stats)
        except Exception as e:
            analyzer.logger.error(f"Failed to analyze corpus '{corpus}': {str(e)}")
