# README

## Overview

This project implements a computational method for estimating the entropy and redundancy of a language, inspired by the techniques described by Claude E. Shannon in his seminal paper, "Prediction and Entropy of Printed English" (1951). While Shannon's original work focused on the English language, this project extends the methodology to analyze the Linear B script.

## Background

In his 1951 paper, Shannon introduced a method to estimate the entropy and redundancy of a language by leveraging the predictability of characters based on preceding text. Entropy is a measure of the average information content per symbol, while redundancy indicates the degree to which a language constrains or structures its text, making some sequences more predictable than others.

Shannon's experiments involved predicting the next letter in a sequence of text, and he used this predictability to calculate entropy. This project replicates Shannon's methodology using modern computational tools and applies it to the Linear B script, an ancient script used primarily for writing Mycenaean Greek.

## Project Structure

- `shannon_entro.py`: The main script that performs entropy and redundancy calculations on various English corpora sourced through NLTK.
- `linearb_entro.py`: A secondary script that performs entropy and redundancy calculations on Linear B.
- `Linear_B_Lexicon.csv`: The input corpus file containing Linear B words.
- `entropy_model/`: Directory where trained KenLM models are stored.
- `README.md`: This document.

## Dependencies

- Python 3.11+
- pandas
- regex
- kenlm (KenLM language model library)
- nltk

## Setup

1. **Install Dependencies**:
   ```bash
   pip install pandas regex kenlm nltk
   ```

2. **Download NLTK Data**:
   If using any NLTK corpora, ensure they are downloaded:
   ```python
   import nltk
   nltk.download('corpus_name')
   ```

3. **Prepare Corpus**:
   Place the `Linear_B_Lexicon.csv` file in the project directory. This file should contain the Linear B words to be analyzed.

## Usage

1. **Run the Script**:
   Execute the main script to process the corpus and calculate entropy and redundancy:
   ```bash
   python linearb_entro.py
   ```

2. **Output**:
   The script will log the results, including:
   - Vocabulary count
   - Alphabet size
   - Zero-order approximation (H0)
   - First-order approximation (H1)
   - Third-order approximation (H3)
   - Redundancy percentage

## Methodology

1. **Load and Format Corpus**:
   The script reads the `Linear_B_Lexicon.csv` file and formats the words by removing duplicates and cleaning the text using regular expressions to match Linear B glyphs.

2. **Build KenLM Model**:
   A KenLM language model is trained on the formatted corpus. This model is used to calculate the entropy based on n-grams (sequences of n adjacent glyphs).

3. **Calculate Entropy**:
   - **H0 (Zero-order approximation)**: Calculated using the logarithm of the alphabet size.
   - **H1 (First-order approximation)**: Calculated using the frequencies of individual glyphs.
   - **H3 (Third-order approximation)**: Calculated using the KenLM model to predict the next glyph in a sequence.

4. **Calculate Redundancy**:
   Redundancy is calculated as the percentage reduction in entropy due to the language's statistical structure:
   ```python
   redundancy = (1 - H3 / H0) * 100
   ```

## Example Output

```
Linear B Corpus
Vocab Count: 1000
Alphabet Size: 87
Zero-order approximation (H0): 6.45
First-order approximation (H1): 4.32
Third-order approximation (H3) of 5-grams: 2.87
Redundancy: 55.42%
```

## References

- Shannon, C. E. (1950). Prediction and Entropy of Printed English. Bell System Technical Journal.
- Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to Claude E. Shannon for his groundbreaking work in information theory, which serves as the foundation for this project.