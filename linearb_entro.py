import logging
import math
import subprocess
from collections import Counter
from pathlib import Path
import tempfile

import kenlm
import numpy as np
import pandas as pd
import regex as reg

# Configuration
Q_GRAMS = 6  # KenLM model n-gram level
MODEL_DIR = Path.cwd() / "entropy_model"
CORPUS_PATH = Path.cwd() / "Linear_B_Lexicon.csv"

# Setup
MODEL_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def ensure_directory_exists(directory_path):
    """Ensure the specified directory exists, creating it if necessary."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def run_command(command, error_message):
    """Run a shell command using subprocess, capturing and logging any errors."""
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()} (Exit code: {e.returncode})")
        return False
    return True

def build_kenlm_model(text, model_directory, q_gram):
    """Build a KenLM language model from the specified text."""
    ensure_directory_exists(model_directory)
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_text_file:
        temp_text_file.write(text.encode('utf-8'))
        temp_text_file_path = temp_text_file.name

    corpus_name = Path(temp_text_file_path).stem
    arpa_file = model_directory / f"{corpus_name}_{q_gram}gram.arpa"
    binary_file = model_directory / f"{corpus_name}_{q_gram}gram.klm"

    arpa_command = f"lmplz -o {q_gram} --text {temp_text_file_path} --arpa {arpa_file} --discount_fallback"
    binary_command = f"build_binary {arpa_file} {binary_file}"

    if run_command(arpa_command, "Failed to generate ARPA model") and run_command(binary_command, "Failed to convert ARPA model to binary format"):
        Path(temp_text_file_path).unlink(missing_ok=True)
        return binary_file
    else:
        Path(temp_text_file_path).unlink(missing_ok=True)
        return None

def load_and_format_corpus(csv_path):
    """Load and format the Linear B corpus data."""
    df = pd.read_csv(csv_path)
    unique_words_series = df['word'].drop_duplicates()

    formatted_words = []
    for word in unique_words_series:
        formatted_word = ' '.join(reg.findall(r'[\U00010000-\U0001007F\U00010080-\U000100FF]', word))
        if (formatted_word := formatted_word.strip()):
            formatted_words.append(formatted_word)

    formatted_text = '\n'.join(formatted_words)
    unique_words = len(formatted_words)

    return formatted_text, unique_words

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
    # Join all tokens and remove spaces to consider character distribution
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

def process_linearb_corpus(corpus_path, q_gram):
    """Process the Linear B corpus to compute entropy and redundancy metrics."""
    formatted_text, unique_words = load_and_format_corpus(corpus_path)
    
    if len(formatted_text) < q_gram:
        logging.error("Insufficient data to build a language model. Increase the data size or decrease the n-gram level.")
        return

    model_path = build_kenlm_model(formatted_text, MODEL_DIR, q_gram)

    if model_path:
        model = kenlm.Model(str(model_path))
        lines = formatted_text.split('\n')
        H0 = math.log2(len(set(''.join(lines).replace(' ', ''))))
        letter_freq = Counter(''.join(lines).replace(' ', ''))
        H1 = calculate_unigram_entropy(formatted_text)
        H2 = calculate_H2(formatted_text)
        H3_kenlm = calculate_entropy_kenlm(model, lines)
        redundancy = calculate_redundancy(H3_kenlm, H0)

        logging.info("Linear B Corpus")
        logging.info(f"Vocab Count: {unique_words}")
        logging.info(f'Grapheme Inventory Size: {len(letter_freq):,}')
        logging.info(f"Zero-order Entropy (H0): {H0:.2f}")
        logging.info(f"First-order Entropy (H1): {H1:.2f}")
        logging.info(f"Second-order Entropy (H2): {H2:.2f}")
        logging.info(f"Third-order Entropy (H3) of {Q_GRAMS}-grams: {H3_kenlm:.2f}")
        logging.info(f"Redundancy: {redundancy:.2f}%")

        # Delete the model file after use
        try:
            Path(model_path).unlink(missing_ok=True)
        except Exception as e:
            logging.error(f"Failed to delete model file: {model_path}, error: {e}")

    else:
        logging.error(f"Failed to process corpus: {corpus_path.stem}")

if __name__ == '__main__':
    process_linearb_corpus(CORPUS_PATH, Q_GRAMS)