import logging
import math
import subprocess
from collections import Counter
from pathlib import Path
import tempfile
import unicodedata

import kenlm
import numpy as np
import pandas as pd
import regex as reg
import nltk
from nltk.corpus import europarl_raw, brown

# Configuration
Q_GRAMS = 8  # KenLM model n-gram level
MODEL_DIR = Path.cwd() / "entropy_model"
CORPUS_PATH = Path.cwd() / "Linear_B_Lexicon.csv"

LANGUAGES = ['english', 'french', 'german', 'italian', 'greek', 'spanish', 'dutch']
CORPORA = ['linear_b', 'brown'] + LANGUAGES

# Setup
MODEL_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')  # Change to DEBUG for detailed logs

# Ensure the necessary NLTK corpora are downloaded
nltk.download('europarl_raw', quiet=True)
nltk.download('brown', quiet=True)

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

def is_letter(char):
    """Check if a character is a letter, including Unicode letters like Linear B."""
    return unicodedata.category(char).startswith('L')

def get_letter_filter(corpus_name):
    """
    Returns a filtering function for letters based on the corpus name.
    
    - English: Only a-z letters.
    - French: a-z plus accented letters like à, â, ç, é, è, ê, ë, î, ï, ô, û, ù, ü, ÿ, ñ, æ, œ.
    - German: a-z plus ä, ö, ü, ß.
    - Italian: a-z plus accented letters like à, è, é, ì, í, î, ò, ó, ù, ú.
    - Spanish: a-z plus ñ and accented vowels á, é, í, ó, ú, ü.
    - Dutch: a-z plus occasional accented letters like ë, ï.
    - Greek: Greek letters (both uppercase and lowercase).
    - Linear B: All letters in the Unicode range U+10000 to U+100FF.
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
    
    # Normalize corpus name to lowercase to ensure case-insensitive matching
    corpus_name_lower = corpus_name.lower()
    
    # Retrieve the allowed letters for the specified corpus
    letters = allowed_letters.get(corpus_name_lower)
    
    if letters is None:
        # Default to all Unicode letters if the corpus is unrecognized
        return is_letter
    else:
        # Return a filter function that checks if the character is in the allowed set
        return lambda char: char in letters


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

def load_and_format_linearb_corpus(csv_path):
    """Load and format the Linear B corpus data."""
    df = pd.read_csv(csv_path)
    unique_words_series = df['word'].drop_duplicates()

    formatted_letters = []
    all_chars = []  # To collect all characters for unique_letters

    for word in unique_words_series:
        # Keep only Linear B characters and include spaces between them
        linear_b_chars = reg.findall(r'[\U00010000-\U0001007F\U00010080-\U000100FF]', word)
        if linear_b_chars:
            # Join letters with space to treat each as a separate token
            formatted_letter = ' '.join(linear_b_chars)
            formatted_letters.append(formatted_letter)
            all_chars.extend(linear_b_chars)  # Collect characters

    # Join all letters across words with newline to preserve word boundaries
    formatted_text = '\n'.join(formatted_letters)
    
    # Unique Letters Count
    unique_letters = len(set(all_chars))

    # Vocab Count as number of unique words
    vocab_count = len(unique_words_series)

    return formatted_text, vocab_count, unique_letters

def load_and_format_brown_corpus():
    """Load and format the Brown corpus data."""
    corpus_sents = brown.sents()
    formatted_letters = []
    all_chars = []  # To collect all characters for unique_letters

    for sent in corpus_sents:
        for word in sent:
            # Treat each letter as a separate token
            letters = ' '.join(list(word))
            formatted_letters.append(letters)
            all_chars.extend(list(word))  # Collect letters

    # Join all letters across words with newline to preserve word boundaries
    formatted_text = '\n'.join(formatted_letters)
    
    # Unique Letters Count
    unique_letters = len(set(all_chars))
    
    # Vocab Count as number of unique words (case-insensitive)
    vocab_count = len(set(word.lower() for word in brown.words()))
    
    return formatted_text, vocab_count, unique_letters

def load_and_format_europarl_corpus(language):
    """Load and format the Europarl corpus data for the specified language."""
    try:
        corpus_sents = getattr(europarl_raw, language).sents()
    except AttributeError:
        logging.error(f"Language '{language}' not found in europarl_raw corpus.")
        return "", 0, 0

    formatted_letters = []
    all_chars = []  # To collect all characters for unique_letters

    for sent in corpus_sents:
        for word in sent:
            # Treat each letter as a separate token
            letters = ' '.join(list(word))
            formatted_letters.append(letters)
            all_chars.extend(list(word))  # Collect letters

    # Join all letters across words with newline to preserve word boundaries
    formatted_text = '\n'.join(formatted_letters)
    
    # Unique Letters Count
    unique_letters = len(set(all_chars))
    
    # Vocab Count as number of unique words
    vocab_count = len(set(word for sent in corpus_sents for word in sent))
    
    return formatted_text, vocab_count, unique_letters

def calculate_entropy_kenlm(model, text):
    """Calculate the entropy of the text using the KenLM model."""
    if isinstance(text, list):
        text = '\n'.join(text)  # Join lines with newline to preserve word boundaries
    
    log_prob = model.score(text, bos=False, eos=False) / math.log(2)
    num_grams = max(len(text.split()) - Q_GRAMS + 1, 1)  # Correct formula for n-grams
    return -log_prob / num_grams

def calculate_unigram_entropy(text, letter_filter, corpus_name):
    """
    Calculate the first-order entropy (unigram entropy) of the text.
    
    Parameters:
    - text (str): The text to analyze.
    - letter_filter (callable): Function to filter valid letters.
    - corpus_name (str): Name of the corpus being processed.
    
    Returns:
    - float: Calculated unigram entropy.
    """
    # Keep only letters based on the provided filter
    text = ''.join(filter(letter_filter, text))
    
    # Convert to lowercase if not Linear B
    if corpus_name.lower() != 'linear_b':
        text = text.lower()
    
    # Count letter frequencies
    unigram_freq = Counter(text)
    total_unigrams = sum(unigram_freq.values())
    
    # Calculate probabilities
    if total_unigrams > 0:
        probabilities = np.array(list(unigram_freq.values())) / total_unigrams
    else:
        probabilities = np.array([])
    
    # Calculate entropy
    if probabilities.size > 0:
        entropy = -np.sum(probabilities * np.log2(probabilities))
    else:
        entropy = 0
    
    return entropy

def calculate_H2(text, letter_filter, corpus_name):
    """
    Calculate the Rényi entropy of order 2 (H2).
    
    Parameters:
    - text (str): The text to analyze.
    - letter_filter (callable): Function to filter valid letters.
    - corpus_name (str): Name of the corpus being processed.
    
    Returns:
    - float: Calculated H2 value in bits.
    """
    # Keep only letters based on the provided filter
    filtered_text = ''.join(filter(letter_filter, text))
    
    # Convert to lowercase if not Linear B
    if corpus_name.lower() != 'linear_b':
        filtered_text = filtered_text.lower()
    
    # Count letter frequencies
    char_freq = Counter(filtered_text)
    total_chars = len(filtered_text)
    
    # Calculate probabilities
    if total_chars > 0:
        probabilities = np.array(list(char_freq.values())) / total_chars
        H2 = -np.log2(np.sum(probabilities**2))
    else:
        H2 = 0
    
    return H2

def calculate_redundancy(H, H_max):
    """Calculate the redundancy of the text."""
    return (1 - H / H_max) * 100 if H_max > 0 else 0

def process_corpus(corpus_name, q_gram):
    """Process a corpus to compute entropy and redundancy metrics."""
    logging.info(f"\nProcessing {corpus_name.replace('_', ' ').title()} Corpus")
    
    if corpus_name == 'linear_b':
        formatted_text, vocab_count, unique_letters = load_and_format_linearb_corpus(CORPUS_PATH)
        lowercase = False
    elif corpus_name == 'brown':
        formatted_text, vocab_count, unique_letters = load_and_format_brown_corpus()
        lowercase = True
    else:
        formatted_text, vocab_count, unique_letters = load_and_format_europarl_corpus(corpus_name)
        lowercase = True
    
    if len(formatted_text.split()) < q_gram:
        logging.error(f"Insufficient data in {corpus_name.title()} corpus to build a language model.")
        return
    
    model_path = build_kenlm_model(formatted_text, MODEL_DIR, q_gram, corpus_name)
    
    if model_path:
        model = kenlm.Model(str(model_path))
        lines = formatted_text.split('\n')
        
        # Get the appropriate letter filter
        letter_filter = get_letter_filter(corpus_name)
        
        # Keep only letters for Grapheme Inventory and H0 calculation
        text_letters = ''.join(filter(letter_filter, ''.join(lines)))
        if lowercase:
            text_letters = text_letters.lower()
        unique_letters_set = set(text_letters)
        letter_freq = Counter(text_letters)
        
        H_max = math.log2(len(unique_letters_set)) if len(unique_letters_set) > 0 else 0
        H1 = calculate_unigram_entropy(formatted_text, letter_filter, corpus_name)
        H2 = calculate_H2(formatted_text, letter_filter, corpus_name)
        H3_kenlm = calculate_entropy_kenlm(model, lines)  # Correct H3 calculation
        redundancy = calculate_redundancy(H3_kenlm, H_max) if H_max > 0 else 0
        
        logging.info(f"Vocab Count: {vocab_count}")  # Number of unique words
        logging.info(f'Grapheme Inventory: {len(unique_letters_set)}')  # Number of unique letters
        logging.info(f"Zero-order Entropy (H0): {H_max:.2f}")
        logging.info(f"First-order Entropy (H1): {H1:.2f}")
        logging.info(f"Second-order Entropy (H2): {H2:.2f}")
        logging.info(f"Third-order Entropy (H3) of {Q_GRAMS}-grams: {H3_kenlm:.2f}")
        logging.info(f"Redundancy: {redundancy:.2f}%")
        
        # Optional: Log unique letters for debugging
        # logging.debug(f"Unique Letters in {corpus_name}: {unique_letters_set}")
        
        # Delete the model files after use
        try:
            Path(model_path).unlink(missing_ok=True)
            arpa_file = model_path.with_suffix('.arpa')
            arpa_file.unlink(missing_ok=True)
        except Exception as e:
            logging.error(f"Failed to delete model files for {corpus_name}, error: {e}")
        
        return redundancy
    else:
        logging.error(f"Failed to process corpus: {corpus_name.title()}")
        return None

if __name__ == '__main__':
    redundancies = {}
    for corpus in CORPORA:
        redundancy = process_corpus(corpus, Q_GRAMS)
        if redundancy is not None:
            redundancies[corpus.replace('_', ' ').title()] = redundancy
    
    if redundancies:
        logging.info("\nRedundancy Rates Across Corpora:")
        for corpus_name, redundancy in redundancies.items():
            logging.info(f"{corpus_name}: {redundancy:.2f}%")
