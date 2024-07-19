import logging
import math
import subprocess
from collections import Counter
from pathlib import Path
import tempfile

import kenlm
import pandas as pd
import regex

# Configuration
CORPUS_PATH = Path.cwd() / "Linear_B_Lexicon.csv"
MODEL_DIR = Path.cwd() / "entropy_model"
Q_GRAMS = 5  # KenLM model n-gram level

# Setup
MODEL_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def run_command(command, error_message):
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()} (Exit code: {e.returncode})")
        return False
    return True


def build_kenlm_model(text, model_directory, q_gram):
    with tempfile.NamedTemporaryFile(delete=False) as temp_text_file:
        temp_text_file.write(text.encode('utf-8'))
        temp_text_file_path = temp_text_file.name

    with tempfile.NamedTemporaryFile(delete=False) as temp_arpa_file:
        temp_arpa_file_path = temp_arpa_file.name

    corpus_name = Path(temp_text_file_path).stem
    binary_file = model_directory / f"{corpus_name}_{q_gram}gram.klm"

    if run_command(f"lmplz -o {q_gram} --text {temp_text_file_path} --arpa {temp_arpa_file_path}", "Failed to generate ARPA model") and \
       run_command(f"build_binary {temp_arpa_file_path} {binary_file}", "Failed to convert ARPA model to binary format"):
        Path(temp_text_file_path).unlink(missing_ok=True)
        Path(temp_arpa_file_path).unlink(missing_ok=True)
        return binary_file
    else:
        Path(temp_text_file_path).unlink(missing_ok=True)
        Path(temp_arpa_file_path).unlink(missing_ok=True)
        return None


def load_and_format_corpus(csv_path):
    df = pd.read_csv(csv_path)
    unique_words_series = df['word'].drop_duplicates()

    formatted_words = []
    for word in unique_words_series:
        formatted_word = ' '.join(regex.findall(r'[\U00010000-\U0001007F\U00010080-\U000100FF]', word))
        if formatted_word.strip():
            formatted_words.append(formatted_word)

    formatted_text = '\n'.join(formatted_words)
    unique_words = len(formatted_words)

    return formatted_text, unique_words


def calculate_entropy_kenlm(model, lines):
    prepared_text = ' '.join(lines)
    log_prob = model.score(prepared_text, bos=False, eos=False)
    log_prob /= math.log(2)
    num_grams = max(len(prepared_text.split()) - Q_GRAMS, 1)  # Prevent division by zero
    return -log_prob / num_grams


def calculate_redundancy(H, H_max):
    return (1 - H / H_max) * 100


def process_linearb_corpus(corpus_path, q_gram):
    formatted_text, unique_words = load_and_format_corpus(corpus_path)
    model_path = build_kenlm_model(formatted_text, MODEL_DIR, q_gram)

    if model_path:
        model = kenlm.Model(str(model_path))
        lines = formatted_text.split('\n')
        H0 = math.log2(len(set(''.join(lines).replace(' ', ''))))
        letter_freq = Counter(''.join(lines).replace(' ', ''))
        total_letters = sum(letter_freq.values())
        H1 = -sum((freq / total_letters) * math.log2(freq / total_letters) for freq in letter_freq.values())
        H3_kenlm = calculate_entropy_kenlm(model, lines)
        redundancy = calculate_redundancy(H3_kenlm, H0)

        logging.info("Linear B Corpus")
        logging.info(f"Vocab Count: {unique_words}")
        logging.info(f'Alphabet Size: {len(letter_freq):,}')
        logging.info(f"Zero-order approximation (H0): {H0:.2f}")
        logging.info(f"First-order approximation (H1): {H1:.2f}")
        logging.info(f"Third-order approximation (H3) of {Q_GRAMS}-grams: {H3_kenlm:.2f}")
        logging.info(f"Redundancy: {redundancy:.2f}%")

if __name__ == '__main__':
    process_linearb_corpus(CORPUS_PATH, Q_GRAMS)