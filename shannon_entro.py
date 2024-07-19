import logging
import math
import subprocess
from collections import Counter
from pathlib import Path
import tempfile

import kenlm
import numpy as np
import nltk
import regex as reg

# Configuration
Q_GRAMS = 6  # KenLM model n-gram level
MODEL_DIR = Path.cwd() / "entropy_model"

# Setup
MODEL_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def ensure_directory_exists(directory_path):
    """Ensure the specified directory exists, creating it if necessary."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def ensure_corpus_available(corpus_name):
    """Ensure the specified NLTK corpus is available for use."""
    nltk.download(corpus_name, quiet=True)


def clean_and_format_words(words):
    """Clean and format words by removing non-letter characters and converting to lowercase."""
    return [' '.join(reg.sub(r'[^\p{L}]', '', word).lower()) for word in words if len(word) >= 3]


def run_command(command, error_message):
    """Run a shell command using subprocess, capturing and logging any errors."""
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()} (Exit code: {e.returncode})")
        return False
    return True


def build_kenlm_model(text, model_directory, q_gram):
    ensure_directory_exists(model_directory)
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_text_file:
        temp_text_file.write(text.encode('utf-8'))
        temp_text_file_path = temp_text_file.name

    with tempfile.NamedTemporaryFile(delete=False) as temp_arpa_file:
        temp_arpa_file_path = temp_arpa_file.name

    corpus_name = Path(temp_text_file_path).stem
    binary_file = model_directory / f"{corpus_name}_{q_gram}gram.klm"

    if run_command(f"lmplz -o {q_gram} --discount_fallback --text {temp_text_file_path} --arpa {temp_arpa_file_path}", "Failed to generate ARPA model") and \
       run_command(f"build_binary {temp_arpa_file_path} {binary_file}", "Failed to convert ARPA model to binary format"):
        Path(temp_text_file_path).unlink(missing_ok=True)
        Path(temp_arpa_file_path).unlink(missing_ok=True)
        return binary_file
    else:
        Path(temp_text_file_path).unlink(missing_ok=True)
        Path(temp_arpa_file_path).unlink(missing_ok=True)
        return None


def calculate_entropy_kenlm(model, text):
    """Calculate the entropy of the text using the KenLM model."""
    if isinstance(text, list):
        text = ' '.join(text)
    
    log_prob = model.score(text, bos=False, eos=False) / math.log(2)
    num_grams = max(len(text.split()) - Q_GRAMS, 1)  # Prevent division by zero
    return -log_prob / num_grams


def calculate_unigram_entropy(text):
    """Calculate the first-order entropy (unigram entropy) of the text."""
    # Count character frequencies
    unigram_freq = Counter(text.replace('\n', '').replace(' ', ''))
    
    # Convert frequencies to probabilities
    total_unigrams = sum(unigram_freq.values())
    probabilities = np.array(list(unigram_freq.values())) / total_unigrams
    
    # Calculate entropy
    return -np.sum(probabilities * np.log2(probabilities))


def calculate_H2(text):
    """
    Calculate the Rényi entropy of order 2 (H2).
    This is also known as collision entropy.
    """
    text = text.replace(' ', '')
    
    # Count character frequencies
    char_freq = Counter(text)
    total_chars = len(text)
    
    # Calculate probabilities
    probabilities = np.array([count / total_chars for count in char_freq.values()])
    
    # Calculate H2
    H2 = -np.log2(np.sum(probabilities**2))
    
    return H2


def calculate_redundancy(H, H_max):
    """Calculate the redundancy of the text."""
    return (1 - H / H_max) * 100


def process_single_corpus(corpus_name):
    """Process a single NLTK corpus to compute entropy and redundancy metrics."""
    ensure_corpus_available(corpus_name)
    words = getattr(nltk.corpus, corpus_name).words()

    cleaned_words = clean_and_format_words(words)
    text_for_kenlm = '\n'.join(cleaned_words)

    model_path = build_kenlm_model(text_for_kenlm, MODEL_DIR, Q_GRAMS)
    if model_path:
        model = kenlm.Model(str(model_path))

        alphabet = sorted(set(text_for_kenlm.replace('\n', '').replace(' ', '')))
        H0 = math.log2(len(alphabet))
        H3_kenlm = calculate_entropy_kenlm(model, text_for_kenlm)
        redundancy = calculate_redundancy(H3_kenlm, H0)
        H1 = calculate_unigram_entropy(text_for_kenlm)
        H2 = calculate_H2(text_for_kenlm)

        logging.info(f"\nCorpus: {corpus_name}")
        logging.info(f"Token Count: {len(words)}")
        logging.info(f"Vocab Count: {len(set(words))}")
        logging.info(f"Alphabet Size: {len(alphabet)}")
        logging.info(f"Zero-order approximation (H0): {H0:.2f}")
        logging.info(f"First-order approximation (H1): {H1:.2f}")
        logging.info(f"Second-order approximation (H2): {H2:.2f}")
        logging.info(f"Third-order approximation (H3) of {Q_GRAMS}-grams: {H3_kenlm:.2f}")
        logging.info(f"Redundancy: {redundancy:.2f}%")
    else:
        logging.error(f"Failed to process corpus: {corpus_name}")


def process_corpora(corpus_list):
    """Process a list of corpora to compute entropy and redundancy metrics for each."""
    for corpus_name in corpus_list:
        process_single_corpus(corpus_name)


# Execute the main function
if __name__ == "__main__":
    process_corpora(['brown', 'reuters', 'webtext', 'inaugural', 'nps_chat', 'state_union', 'gutenberg'])