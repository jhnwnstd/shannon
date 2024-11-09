import logging
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import sys

import kenlm  # Ensure KenLM is installed: pip install https://github.com/kpu/kenlm/archive/master.zip
import numpy as np
import regex  # Ensure using the third-party regex module: pip install regex
import nltk
from nltk.corpus import *
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ============================
# Configuration
# ============================

Q_GRAMS = 8  # KenLM model n-gram level
MODEL_DIR = Path.cwd() / "entropy_model"

# Mapping from language codes to language names
LANGUAGE_CODE_MAP = {
    'bg': 'bulgarian',
    'cs': 'czech',
    'da': 'danish',
    'de': 'german',
    'el': 'greek',
    'en': 'english',
    'es': 'spanish',
    'et': 'estonian',
    'fi': 'finnish',
    'fr': 'french',
    'hu': 'hungarian',
    'it': 'italian',
    'lt': 'lithuanian',
    'lv': 'latvian',
    'nl': 'dutch',
    'pl': 'polish',
    'pt': 'portuguese',
    'ro': 'romanian',
    'sk': 'slovak',
    'sl': 'slovene',
    'sv': 'swedish',
    # 'ru': 'russian',  # Uncomment if Russian is available
    # Add more language codes as needed
}

# List of standard NLTK corpora to process
STANDARD_CORPORA = ['brown', 'reuters', 'webtext', 'inaugural', 'nps_chat', 'state_union', 'gutenberg']

# ============================
# Setup
# ============================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure required corpora are downloaded
required_corpora = STANDARD_CORPORA + ['europarl_raw']
for corpus in required_corpora:
    try:
        nltk.data.find(f'corpora/{corpus}')
    except LookupError:
        logging.info(f"Downloading {corpus} corpus...")
        nltk.download(corpus)

# ============================
# Helper Functions and Data Classes
# ============================

@dataclass
class EntropyResults:
    """Container for entropy calculation results"""
    h0: float  # Maximum possible entropy
    h1: float  # First-order entropy
    h2: float  # Second-order entropy
    h3: float  # Third-order entropy (KenLM-based)
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

def ensure_directory_exists(directory_path: Path) -> None:
    """Ensure the specified directory exists, creating it if necessary."""
    directory_path.mkdir(parents=True, exist_ok=True)

def run_command(command: str, error_message: str) -> bool:
    """
    Run a shell command using subprocess, capturing and logging any errors.
    
    Parameters:
    - command (str): The command to execute.
    - error_message (str): The error message to log if the command fails.
    
    Returns:
    - bool: True if the command succeeds, False otherwise.
    """
    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode().strip()} (Exit code: {e.returncode})")
        return False
    return True

def get_letter_filter(language_name: str) -> callable:
    """
    Returns a filtering function for letters based on the language name.
    
    Parameters:
    - language_name (str): Name of the language.
    
    Returns:
    - callable: Function that takes a character and returns True if it should be included.
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
        'danish': set('abcdefghijklmnopqrstuvwxyzæøåABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ'),
        'portuguese': set('abcdefghijklmnopqrstuvwxyzçáàãâéêíóôõúABCDEFGHIJKLMNOPQRSTUVWXYZÇÁÀÃÂÉÊÍÓÔÕÚ'),
        'romanian': set('abcdefghijklmnopqrstuvwxyzăâîșțțABCDEFGHIJKLMNOPQRSTUVWXYZĂÂÎȘȚ'),
        'slovak': set('abcdefghijklmnopqrstuvwxyzáčďéíľĺňóôŕšťúýžABCDEFGHIJKLMNOPQRSTUVWXYZÁČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ'),
        'slovene': set('abcdefghijklmnopqrstuvwxyzčšžABCDEFGHIJKLMNOPQRSTUVWXYZČŠŽ'),
        'swedish': set('abcdefghijklmnopqrstuvwxyzåäöABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ'),
        # Add more languages as needed
    }
    
    letters = allowed_letters.get(language_name.lower())
    
    if letters is None:
        # Default to all Unicode letters if the language is unrecognized
        return lambda char: regex.match(r'\p{L}', char) is not None
    else:
        # Return a filter function that checks if the character is in the allowed set
        return lambda char: char in letters

def clean_and_format_words(words: List[str], language_name: str) -> List[str]:
    """
    Clean and format words by removing non-letter characters, converting to lowercase, and separating letters with spaces.
    
    Parameters:
    - words (List[str]): List of words from the corpus.
    - language_name (str): Name of the language being processed.
    
    Returns:
    - List[str]: Cleaned and formatted words.
    """
    cleaned_words = []
    letter_filter = get_letter_filter(language_name)
    for word in words:
        try:
            # Remove non-letter characters using regex
            cleaned_word = regex.sub(r'[^\p{L}]', '', word)
            if len(cleaned_word) >= 3:
                # Convert to lowercase if not Linear B
                if language_name.lower() != 'linear_b':
                    cleaned_word = cleaned_word.lower()
                # Filter letters
                filtered_letters = ''.join([char for char in cleaned_word if letter_filter(char)])
                if filtered_letters:
                    # Treat each letter as a separate token by joining with spaces
                    formatted_word = ' '.join(filtered_letters)
                    cleaned_words.append(formatted_word)
        except regex.error as regex_err:
            logging.error(f"Regex error while processing word '{word}': {regex_err}")
    return cleaned_words

def build_kenlm_model(text: str, model_directory: Path, q_gram: int, corpus_name: str) -> Optional[Path]:
    """
    Build a KenLM language model from the specified text.
    
    Parameters:
    - text (str): The corpus text to build the model from.
    - model_directory (Path): Directory to store the model files.
    - q_gram (int): The n-gram order.
    - corpus_name (str): Name of the corpus.
    
    Returns:
    - Optional[Path]: Path to the binary KenLM model if successful, None otherwise.
    """
    ensure_directory_exists(model_directory)
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as temp_text_file:
        temp_text_file.write(text)
        temp_text_file_path = temp_text_file.name

    arpa_file = model_directory / f"{corpus_name}_{q_gram}gram.arpa"
    binary_file = model_directory / f"{corpus_name}_{q_gram}gram.klm"

    # **[CORRECTION]** Quote the file paths to handle special characters like parentheses
    arpa_command = f"lmplz -o {q_gram} --text \"{temp_text_file_path}\" --arpa \"{arpa_file}\" --discount_fallback"
    binary_command = f"build_binary \"{arpa_file}\" \"{binary_file}\""

    if run_command(arpa_command, "Failed to generate ARPA model") and run_command(binary_command, "Failed to convert ARPA model to binary format"):
        Path(temp_text_file_path).unlink(missing_ok=True)
        return binary_file
    else:
        Path(temp_text_file_path).unlink(missing_ok=True)
        return None

def calculate_entropy_kenlm(model: kenlm.Model, text: str) -> float:
    """
    Calculate the entropy of the text using the KenLM model.
    
    Parameters:
    - model (kenlm.Model): The KenLM language model.
    - text (str): The text to analyze.
    
    Returns:
    - float: Calculated entropy in bits.
    """
    # Calculate the total log probability of the text
    log_prob = model.score(text, bos=False, eos=False)  # log base e
    log_prob_bits = log_prob / math.log(2)  # Convert to log base 2
    
    # Estimate the number of n-grams
    num_tokens = len(text.split())
    num_ngrams = max(num_tokens - Q_GRAMS + 1, 1)  # Prevent division by zero
    
    # Calculate entropy
    entropy = -log_prob_bits / num_ngrams
    return entropy

def calculate_redundancy(H: float, H_max: float) -> float:
    """
    Calculate the redundancy of the text.
    
    Parameters:
    - H (float): Calculated entropy.
    - H_max (float): Maximum possible entropy.
    
    Returns:
    - float: Redundancy percentage.
    """
    return (1 - H / H_max) * 100 if H_max > 0 else 0

# ============================
# ShannonAnalyzer Class
# ============================

class ShannonAnalyzer:
    def __init__(self, ngram_order: int = 8):
        self.ngram_order = ngram_order
        self._setup_logging()
        self._download_corpora()
        ensure_directory_exists(MODEL_DIR)
        
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _download_corpora(self) -> None:
        required = [
            'brown', 'reuters', 'webtext', 'inaugural', 'nps_chat', 
            'state_union', 'gutenberg', 'europarl_raw'
        ]
        for corpus in required:
            try:
                nltk.data.find(f'corpora/{corpus}')
            except LookupError:
                self.logger.info(f"Downloading {corpus} corpus...")
                nltk.download(corpus)

    def preprocess_text(self, text: str, language_name: str) -> List[str]:
        """
        Preprocess text by cleaning, filtering, and formatting.
        
        Parameters:
        - text (str): Raw text from the corpus.
        - language_name (str): Name of the language.
        
        Returns:
        - List[str]: Cleaned and formatted words.
        """
        words = text.split()
        cleaned_formatted_words = clean_and_format_words(words, language_name)
        return cleaned_formatted_words

    def calculate_ngram_stats(self, words: List[str], n: int) -> Tuple[Counter, int]:
        """
        Calculate n-gram frequencies and total number of n-grams.
        
        Parameters:
        - words (List[str]): List of formatted words.
        - n (int): n-gram order.
        
        Returns:
        - Tuple[Counter, int]: n-gram frequency counter and total n-grams count.
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
        
        Parameters:
        - freq (Counter): Frequency counter for n-grams.
        - total (int): Total number of n-grams.
        - prev_freq (Optional[Counter]): Frequency counter for (n-1)-grams.
        - prev_total (Optional[int]): Total number of (n-1)-grams.
        
        Returns:
        - float: Calculated entropy in bits.
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

    def analyze_text(self, formatted_words: List[str], language_name: str) -> EntropyResults:
        """
        Analyze a single text to calculate entropy and related metrics.
        
        Parameters:
        - formatted_words (List[str]): List of cleaned and formatted words.
        - language_name (str): Name of the language.
        
        Returns:
        - EntropyResults: Container with calculated entropy and metrics.
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
        # h3 will be replaced by KenLM-based entropy later
        
        # Calculate distributions (Top 10 for simplicity)
        char_dist = {c: count / total_chars for c, count in char_freq.most_common(10)}
        
        # Calculate transition matrices for digrams
        transitions = defaultdict(lambda: defaultdict(float))
        for digram, count in digram_freq.items():
            first, second = digram[0], digram[1]
            transitions[first][second] = count / char_freq[first] if char_freq[first] > 0 else 0.0
        
        # Calculate advanced metrics
        markov_efficiency = 100 * (h1 - h2) / h1 if h1 > 0 else 0.0
        # compression_ratio and predictability will be based on KenLM's H3
        branching_factor = np.mean([len(trans) for trans in transitions.values()]) if transitions else 0.0
        
        return EntropyResults(
            h0=h0,
            h1=h1,
            h2=h2,
            h3=0.0,  # Placeholder; will be updated with KenLM-based H3
            alphabet_size=len(char_freq),
            unique_digrams=len(digram_freq),
            unique_trigrams=len(trigram_freq),
            total_chars=total_chars,
            char_distribution=char_dist,
            digram_distribution={k: count / total_digrams for k, count in digram_freq.most_common(10)},
            trigram_distribution={k: count / total_trigrams for k, count in trigram_freq.most_common(10)},
            transitions={k: dict(v) for k, v in transitions.items()},
            markov_efficiency=markov_efficiency,
            compression_ratio=0.0,  # Placeholder; will be updated
            predictability=0.0,      # Placeholder; will be updated
            branching_factor=branching_factor,
            char_freq=char_freq,          # Absolute counts
            digram_freq=digram_freq,      # Absolute counts
            trigram_freq=trigram_freq     # Absolute counts
        )

    def analyze_corpus_with_kenlm(self, corpus_name: str, language_code: Optional[str] = None, max_files: Optional[int] = None) -> CorpusStatistics:
        """
        Analyze an entire corpus or a specific language subset to compute entropy and redundancy metrics, incorporating KenLM for H3.
        
        Parameters:
        - corpus_name (str): Name of the corpus to analyze.
        - language_code (Optional[str]): Language code (e.g., 'en', 'de'). Required for multi-language corpora like 'europarl_raw'.
        - max_files (Optional[int]): Maximum number of files to process. Processes all if None.
        
        Returns:
        - CorpusStatistics: Container with aggregated corpus statistics.
        """
        if corpus_name == 'europarl_raw' and not language_code:
            raise ValueError("Please specify a language code for 'europarl_raw' corpus (e.g., 'en', 'de').")
        
        # Determine display name and language name
        if corpus_name == 'europarl_raw':
            language_name = LANGUAGE_CODE_MAP.get(language_code, None)
            if not language_name:
                raise ValueError(f"Unsupported language code '{language_code}' for 'europarl_raw' corpus.")
            display_corpus_name = f"europarl_raw.{language_code} ({language_name.capitalize()})"
        else:
            language_name = corpus_name  # For single-language corpora like 'brown', 'reuters', etc.
            display_corpus_name = corpus_name.capitalize()
        
        self.logger.info(f"Starting analysis of '{display_corpus_name}' corpus...")
        
        # Handle 'europarl_raw' with language_code
        if corpus_name == 'europarl_raw':
            from nltk.corpus import europarl_raw
            # Access the language-specific corpus reader
            corpus_reader = getattr(europarl_raw, language_name)
            file_ids = corpus_reader.fileids()[:max_files] if max_files else corpus_reader.fileids()
        else:
            try:
                corpus = getattr(nltk.corpus, corpus_name)
            except AttributeError:
                raise ValueError(f"Corpus '{corpus_name}' not found in NLTK corpus library.")
            file_ids = corpus.fileids()[:max_files] if max_files else corpus.fileids()
        
        # Check if file_ids is empty
        if not file_ids:
            self.logger.warning(f"No files found for corpus '{display_corpus_name}'.")
            return CorpusStatistics(
                corpus_name=display_corpus_name,
                files_analyzed=0,
                total_chars=0,
                mean_entropy={},
                std_entropy={},
                reductions={},
                patterns={},
                efficiency_metrics={}
            )
        
        results = []
        total_chars = 0
        total_char_freq = Counter()
        total_digram_freq = Counter()
        total_trigram_freq = Counter()
        aggregated_text = ""
        
        # Iterate through each file and analyze
        for file_id in tqdm(file_ids, desc=f"Analyzing '{display_corpus_name}' files"):
            try:
                # Retrieve raw text from the file
                if corpus_name == 'europarl_raw':
                    text = corpus_reader.raw(file_id)
                else:
                    text = corpus.raw(file_id)
                
                # Preprocess the text
                formatted_words = self.preprocess_text(text, language_name)
                
                # Analyze the formatted words
                result = self.analyze_text(formatted_words, language_name)
                results.append(result)
                total_chars += result.total_chars
                
                # Aggregate absolute character counts
                total_char_freq.update(result.char_freq)
                
                # Aggregate absolute digram and trigram counts
                total_digram_freq.update(result.digram_freq)
                total_trigram_freq.update(result.trigram_freq)
                
                # Prepare text for KenLM (join formatted words with newline to preserve word boundaries)
                aggregated_text += '\n'.join(formatted_words) + '\n'
            except Exception as e:
                self.logger.warning(f"Error processing '{file_id}': {str(e)}")
        
        if total_chars == 0:
            self.logger.warning(f"No characters found in corpus '{display_corpus_name}'.")
            return CorpusStatistics(
                corpus_name=display_corpus_name,
                files_analyzed=len(results),
                total_chars=total_chars,
                mean_entropy={},
                std_entropy={},
                reductions={},
                patterns={},
                efficiency_metrics={}
            )
        
        # Calculate aggregated character distribution (Top 10)
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
        # h3 will be updated with KenLM-based entropy
        
        # Calculate advanced metrics based on aggregated entropies
        markov_efficiency = 100 * (h1 - h2) / h1 if h1 > 0 else 0.0
        # compression_ratio and predictability will be based on KenLM's H3
        # branching_factor remains the same
        
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
            # 'h3' will be set after KenLM calculation
        }
        std_entropy = {
            'h0': np.std([r.h0 for r in results]),
            'h1': np.std([r.h1 for r in results]),
            'h2': np.std([r.h2 for r in results]),
            # 'h3' will be set after KenLM calculation
        }
        
        # Calculate reductions up to h2
        reductions = {
            'h0_to_h1': 100 * (1 - mean_entropy['h1'] / mean_entropy['h0']) if mean_entropy['h0'] > 0 else 0.0,
            'h1_to_h2': 100 * (1 - mean_entropy['h2'] / mean_entropy['h1']) if mean_entropy['h1'] > 0 else 0.0,
            # 'h2_to_h3' and 'total' will be set after KenLM calculation
        }

        # Aggregate patterns (Top 5 characters)
        patterns = {
            'chars': dict(sorted(char_distribution.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:5])
        }
        
        # Build KenLM model and calculate H3 entropy
        self.logger.info("Building KenLM model for H3 entropy calculation...")
        model_path = build_kenlm_model(aggregated_text, MODEL_DIR, self.ngram_order, display_corpus_name)
        if model_path and model_path.exists():
            try:
                model = kenlm.Model(str(model_path))
                h3_kenlm = calculate_entropy_kenlm(model, aggregated_text)
                redundancy = calculate_redundancy(h3_kenlm, h0)
                self.logger.info(f"KenLM H3 Entropy: {h3_kenlm:.2f} bits")
                self.logger.info(f"Redundancy based on KenLM H3: {redundancy:.2f}%")
            except Exception as e:
                self.logger.error(f"Failed to calculate H3 using KenLM for corpus '{display_corpus_name}': {e}")
                h3_kenlm = 0.0  # Default value if KenLM fails
                redundancy = 0.0
        else:
            self.logger.error(f"KenLM model creation failed for corpus '{display_corpus_name}'. Using trigram entropy for H3.")
            h3_kenlm = 0.0  # Default value if KenLM fails
            redundancy = 0.0
        
        # Update mean and std entropy with KenLM-based H3
        h3_list = [h3_kenlm] * len(results)  # Assuming H3 is same for all files; alternatively, calculate per file
        mean_entropy['h3'] = np.mean(h3_list) if h3_list else 0.0
        std_entropy['h3'] = np.std(h3_list) if h3_list else 0.0
        
        # Update reductions with KenLM-based H3
        reductions.update({
            'h2_to_h3': 100 * (1 - mean_entropy['h3'] / mean_entropy['h2']) if mean_entropy['h2'] > 0 else 0.0,
            'total': 100 * (1 - mean_entropy['h3'] / mean_entropy['h0']) if mean_entropy['h0'] > 0 else 0.0
        })
        
        # Calculate compression_ratio and predictability based on KenLM's H3
        compression_ratio = h3_kenlm / mean_entropy['h0'] if mean_entropy['h0'] > 0 else 0.0
        predictability = 100 * (1 - h3_kenlm / mean_entropy['h0']) if mean_entropy['h0'] > 0 else 0.0
        
        # Update efficiency_metrics
        efficiency_metrics = {
            'markov_efficiency': markov_efficiency,
            'compression_ratio': compression_ratio,
            'predictability': predictability,
            'branching_factor': branching_factor
        }
        
        # **[CORRECTION]** Only attempt to delete model files if model_path is not None
        if model_path:
            try:
                Path(model_path).unlink(missing_ok=True)
                arpa_file = model_path.with_suffix('.arpa')
                arpa_file.unlink(missing_ok=True)
            except Exception as e:
                self.logger.error(f"Failed to delete KenLM model files for '{display_corpus_name}': {e}")
        
        return CorpusStatistics(
            corpus_name=display_corpus_name,
            files_analyzed=len(results),
            total_chars=total_chars,
            mean_entropy=mean_entropy,
            std_entropy=std_entropy,
            reductions=reductions,
            patterns=patterns,
            efficiency_metrics=efficiency_metrics
        )

# ============================
# Analysis Printing Function
# ============================

def print_analysis(stats: CorpusStatistics) -> None:
    """
    Print the analysis results in a formatted manner.
    
    Parameters:
    - stats (CorpusStatistics): The statistics to print.
    """
    print(f"\n{stats.corpus_name} Corpus Analysis")
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
    # Verify regex module is correctly imported
    try:
        import regex
        print(f"Using regex module: {regex.__name__}")
        print(f"Regex module version: {regex.__version__}")
    except ImportError as e:
        print(f"Regex module is not installed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize the analyzer
    analyzer = ShannonAnalyzer(ngram_order=8)
    
    # Define the list of corpora to analyze
    # For multi-language corpora like 'europarl_raw', specify language codes
    CORPORA = [
        ('brown', None),
        ('reuters', None),
        ('webtext', None),
        ('inaugural', None),
        ('nps_chat', None),
        ('state_union', None),
        ('gutenberg', None),
        # Add europarl_raw with different language codes
        ('europarl_raw', 'en'),
        ('europarl_raw', 'de'),
        ('europarl_raw', 'fr'),
        ('europarl_raw', 'es'),
        ('europarl_raw', 'it'),
        ('europarl_raw', 'nl'),
        ('europarl_raw', 'pt'),
        ('europarl_raw', 'sv'),
        ('europarl_raw', 'da'),
        ('europarl_raw', 'fi'),
        ('europarl_raw', 'el'),
        # Add more as needed
    ]
    
    # Analyze each corpus and print results
    for corpus, lang_code in CORPORA:
        try:
            if corpus == 'europarl_raw':
                if not lang_code:
                    raise ValueError("Language code must be specified for 'europarl_raw' corpus.")
                stats = analyzer.analyze_corpus_with_kenlm(corpus, language_code=lang_code, max_files=None)
            else:
                stats = analyzer.analyze_corpus_with_kenlm(corpus, max_files=None)
            print_analysis(stats)
        except Exception as e:
            analyzer.logger.error(f"Failed to analyze corpus '{corpus}' with language code '{lang_code}': {str(e)}")
