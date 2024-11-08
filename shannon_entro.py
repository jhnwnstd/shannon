import logging
import math
import subprocess
from collections import Counter
from pathlib import Path
import tempfile

import kenlm
import numpy as np
import regex as reg
import nltk

# Configuration
Q_GRAMS = 8  # KenLM model n-gram level
MODEL_DIR = Path.cwd() / "entropy_model"

# Setup
MODEL_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')  # Change to DEBUG for detailed logs

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

def build_kenlm_model(text, model_directory, q_gram, corpus_name):
    """Build a KenLM language model from the specified text."""
    ensure_directory_exists(model_directory)
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as temp_text_file:
        temp_text_file.write(text)
        temp_text_file_path = temp_text_file.name

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

def calculate_entropy_kenlm(model, text):
    """Calculate the entropy of the text using the KenLM model."""
    if isinstance(text, list):
        text = ' '.join(text)
    
    log_prob = model.score(text, bos=False, eos=False) / math.log(2)
    num_grams = max(len(text.split()) - Q_GRAMS, 1)  # Prevent division by zero
    return -log_prob / num_grams

def calculate_unigram_entropy(text, corpus_name):
    """Calculate the first-order entropy (unigram entropy) of the text."""
    # Convert to lowercase if not Linear B
    if corpus_name.lower() != 'linear_b':
        text = text.lower()
    
    # Count character frequencies (excluding spaces and newlines)
    unigram_freq = Counter(text.replace(' ', '').replace('\n', ''))
    
    # Calculate probabilities
    total_unigrams = sum(unigram_freq.values())
    probabilities = np.array(list(unigram_freq.values())) / total_unigrams if total_unigrams > 0 else np.array([])
    
    # Calculate entropy
    return -np.sum(probabilities * np.log2(probabilities)) if probabilities.size > 0 else 0

def calculate_H2(text, corpus_name):
    """
    Calculate the Rényi entropy of order 2 (H2).
    
    Parameters:
    - text (str): The text to analyze.
    - corpus_name (str): Name of the corpus being processed.
    
    Returns:
    - float: Calculated H2 value in bits.
    """
    # Convert to lowercase if not Linear B
    if corpus_name.lower() != 'linear_b':
        text = text.lower()
    
    # Remove spaces and newlines
    text = text.replace(' ', '').replace('\n', '')
    
    # Count character frequencies
    char_freq = Counter(text)
    total_chars = len(text)
    
    # Calculate probabilities
    probabilities = np.array([count / total_chars for count in char_freq.values()]) if total_chars > 0 else np.array([])
    
    # Calculate H2
    return -np.log2(np.sum(probabilities**2)) if probabilities.size > 0 else 0

def calculate_redundancy(H, H_max):
    """Calculate the redundancy of the text."""
    return (1 - H / H_max) * 100 if H_max > 0 else 0

def clean_and_format_words(words, corpus_name):
    """
    Clean and format words by removing non-letter characters and converting to lowercase where appropriate.
    
    Parameters:
    - words (list): List of words from the corpus.
    - corpus_name (str): Name of the corpus being processed.
    
    Returns:
    - list: Cleaned and formatted words.
    """
    cleaned_words = []
    for word in words:
        # Remove non-letter characters using regex
        cleaned_word = reg.sub(r'[^\p{L}]', '', word)
        if len(cleaned_word) >= 3:
            # Convert to lowercase if not Linear B
            if corpus_name.lower() != 'linear_b':
                cleaned_word = cleaned_word.lower()
            # Join letters with space to treat each as a separate token
            formatted_word = ' '.join(cleaned_word)
            cleaned_words.append(formatted_word)
    return cleaned_words

def load_and_format_corpus(corpus_name):
    """
    Load and format the specified NLTK corpus.
    
    Parameters:
    - corpus_name (str): Name of the corpus to load.
    
    Returns:
    - str: Formatted text for KenLM.
    - int: Vocab count (number of unique words).
    - int: Grapheme inventory (number of unique letters).
    """
    # Load words from the corpus
    words = getattr(nltk.corpus, corpus_name).words()
    cleaned_words = clean_and_format_words(words, corpus_name)
    text_for_kenlm = '\n'.join(cleaned_words)
    
    # Calculate Grapheme Inventory
    all_chars = ''.join(cleaned_words).replace(' ', '').replace('\n', '')
    unique_letters_set = set(all_chars)
    grapheme_inventory = len(unique_letters_set)
    
    # Calculate Vocab Count (number of unique words)
    vocab_count = len(set(word for word in cleaned_words))
    
    return text_for_kenlm, vocab_count, grapheme_inventory

def process_single_corpus(corpus_name):
    """Process a single NLTK corpus to compute entropy and redundancy metrics."""
    text_for_kenlm, vocab_count, grapheme_inventory = load_and_format_corpus(corpus_name)
    
    if len(text_for_kenlm.split()) < Q_GRAMS:
        logging.error(f"Insufficient data in {corpus_name.title()} corpus to build a language model.")
        return
    
    model_path = build_kenlm_model(text_for_kenlm, MODEL_DIR, Q_GRAMS, corpus_name)
    if model_path:
        model = kenlm.Model(str(model_path))
        
        H_max = math.log2(grapheme_inventory) if grapheme_inventory > 0 else 0
        H1 = calculate_unigram_entropy(text_for_kenlm, corpus_name)
        H2 = calculate_H2(text_for_kenlm, corpus_name)
        H3_kenlm = calculate_entropy_kenlm(model, text_for_kenlm.split('\n'))  # Passing list of lines
        redundancy = calculate_redundancy(H3_kenlm, H_max)
        
        logging.info(f"\nCorpus: {corpus_name}")
        logging.info(f"Token Count: {len(text_for_kenlm.split())}")
        logging.info(f"Vocab Count: {vocab_count}")
        logging.info(f"Grapheme Inventory: {grapheme_inventory}")
        logging.info(f"Zero-order Entropy (H0): {H_max:.2f}")
        logging.info(f"First-order Entropy (H1): {H1:.2f}")
        logging.info(f"Second-order Entropy (H2): {H2:.2f}")
        logging.info(f"Third-order Entropy (H3) of {Q_GRAMS}-grams: {H3_kenlm:.2f}")
        logging.info(f"Redundancy: {redundancy:.2f}%")
        
        # Delete the model files after use
        try:
            Path(model_path).unlink(missing_ok=True)
            arpa_file = model_path.with_suffix('.arpa')
            arpa_file.unlink(missing_ok=True)
        except Exception as e:
            logging.error(f"Failed to delete model files for {corpus_name}, error: {e}")
    else:
        logging.error(f"Failed to process corpus: {corpus_name.title()}")

def process_corpora(corpus_list):
    """Process a list of corpora to compute entropy and redundancy metrics for each."""
    for corpus_name in corpus_list:
        process_single_corpus(corpus_name)

# Execute the main function
if __name__ == "__main__":
    # List of NLTK corpora to process
    CORPORA = ['brown', 'reuters', 'webtext', 'inaugural', 'nps_chat', 'state_union', 'gutenberg']
    
    # Process each corpus
    process_corpora(CORPORA)
    
    # Example of processing a custom text file
    # Replace 'example.txt' with your actual file path
    file_path = "example.txt"
    if Path(file_path).exists():
        with open(file_path, 'r', encoding='utf-8') as file:
            words = file.read().split()
        cleaned_words = clean_and_format_words(words, "custom_text")
        text_for_kenlm = '\n'.join(cleaned_words)
        vocab_count = len(set(cleaned_words))
        grapheme_inventory = len(set(text_for_kenlm.replace(' ', '').replace('\n', '')))
        
        # Calculate H_max
        H_max = math.log2(grapheme_inventory) if grapheme_inventory > 0 else 0
        
        # Calculate entropies
        H1 = calculate_unigram_entropy(text_for_kenlm, "custom_text")
        H2 = calculate_H2(text_for_kenlm, "custom_text")
        
        # Build and load KenLM model
        model_path = build_kenlm_model(text_for_kenlm, MODEL_DIR, Q_GRAMS, "custom_text")
        if model_path:
            model = kenlm.Model(str(model_path))
            H3_kenlm = calculate_entropy_kenlm(model, text_for_kenlm.split('\n'))
            redundancy = calculate_redundancy(H3_kenlm, H_max)
            
            logging.info(f"\nCorpus: Custom Text")
            logging.info(f"Token Count: {len(cleaned_words)}")
            logging.info(f"Vocab Count: {vocab_count}")
            logging.info(f"Grapheme Inventory: {grapheme_inventory}")
            logging.info(f"Zero-order Entropy (H0): {H_max:.2f}")
            logging.info(f"First-order Entropy (H1): {H1:.2f}")
            logging.info(f"Second-order Entropy (H2): {H2:.2f}")
            logging.info(f"Third-order Entropy (H3) of {Q_GRAMS}-grams: {H3_kenlm:.2f}")
            logging.info(f"Redundancy: {redundancy:.2f}%")
            
            # Delete the model files after use
            try:
                Path(model_path).unlink(missing_ok=True)
                arpa_file = model_path.with_suffix('.arpa')
                arpa_file.unlink(missing_ok=True)
            except Exception as e:
                logging.error(f"Failed to delete model files for Custom Text, error: {e}")
    else:
        logging.info(f"File {file_path} not found.")
