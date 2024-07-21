# README

## Overview

This project implements a computational method for estimating the word-level entropy and redundancy of a language, inspired by the techniques described by Claude E. Shannon in his paper, "Prediction and Entropy of Printed English" (1951). While Shannon's original work focused only on English, this project extends the methodology to analyze various English corpora as well as the Linear B script, the writing system for Mycenaean Greek.

## Background

In his 1951 paper, Shannon introduced a method to estimate the entropy and redundancy of a language by leveraging the predictability of characters based on preceding text. Entropy is a measure of the average information content per symbol, while redundancy indicates the degree to which a language constrains or structures its text, making some sequences more predictable than others.

Shannon's experiments involved predicting the next letter in a sequence of text, and he used this predictability to calculate the entropy. Entropy measures the average amount of information produced by each letter of the text. Higher predictability means lower entropy because the next letter can be guessed with higher accuracy, resulting in less new information.

This project replicates Shannon's methodology using the KenLM, a language modeling tool that allows us to build n-gram models to predict the next character in a sequence based on the preceding characters. By applying this methodology, we can analyze and compute the entropy and redundancy of various English corpora and the Linear B script.

## Project Structure

- `entropy_model/`: Directory where trained KenLM models are stored.
- `Linear_B_Lexicon.csv`: The input corpus file containing Linear B words.
- `linearb_entro.py`: A secondary script that performs entropy and redundancy calculations on Linear B.
- `Prediction_and_Entropy_of_Printed_English.pdf`: Shannon's original paper.
- `README.md`: This document.
- `shannon_entro.py`: The main script that performs entropy and redundancy calculations on various English corpora sourced through NLTK.

## Dependencies

- Python 3.11+
- pandas
- regex
- kenlm (KenLM language model)
- nltk
- numpy

## Setup

1. **Install Dependencies**:
   ```bash
   pip install pandas regex nltk numpy
   ```

2. **Install KenLM**:
   Install KenLM by following the instructions on the [KenLM GitHub page](https://github.com/kpu/kenlm):
   ```bash
   pip install https://github.com/kpu/kenlm/archive/master.zip
   ```

3. **Download NLTK Data**:
   If using any NLTK corpora, ensure they are downloaded:
   ```python
   import nltk
   nltk.download('corpus_name')
   ```

4. **Prepare Corpus**:
   Place the `Linear_B_Lexicon.csv` file in the project directory. This file should contain the Linear B words to be analyzed.

## Usage

1. **Run the English Corpora Script**:
   Execute the script to process English corpora and calculate entropy and redundancy:
   ```bash
   python shannon_entro.py
   ```

2. **Run the Linear B Script**:
   Execute the script to process the Linear B corpus and calculate entropy and redundancy:
   ```bash
   python linearb_entro.py
   ```

3. **Output**:
   The scripts will log the results, including:
   - Vocabulary count
   - Grapheme Inventory (Alphabet Size)
   - Zero-order Entropy (H0)
   - First-order Entropy (H1)
   - Second-order Entropy (H2)
   - Third-order Entropy (H3)
   - Redundancy percentage

## Methodology

1. **Load and Format Corpus**:
   The scripts read the respective corpus files and format the words by removing duplicates and cleaning the text using regular expressions to match the appropriate characters (Latin letters for English, Linear B glyphs for Linear B).

2. **Build KenLM Model**:
   A KenLM language model is trained on the formatted corpus. This model is used to calculate the entropy based on n-grams (sequences of n adjacent characters or glyphs).

3. **Calculate Entropy**:
   - **H0 (Zero-order Entropy)**: Calculated using the logarithm of the alphabet size.
   - **H1 (First-order Entropy)**: Calculated using the frequencies of individual characters or glyphs.
   - **H2 (Second-order Entropy)**: Calculated using the probabilities of encountering the same character twice when randomly sampling. This is also known as collision (or Rényi) entropy and is given by the formula:
     ```python
     H2 = -np.log2(np.sum(probabilities**2))
     ```
   - **H3 (Third-order Entropy)**: Calculated using the KenLM model to predict the next character or glyph in a sequence.

4. **Calculate Redundancy**:
   Redundancy is calculated as the percentage reduction in entropy due to the language's statistical structure:
   ```python
   redundancy = (1 - H3 / H0) * 100
   ```

## Example Output

### English Corpus (Brown)
```
Corpus: brown
Token Count: 1,161,192
Vocab Count: 56,057
Grapheme Inventory: 26
Zero-order Entropy (H0): 4.70
First-order Entropy (H1): 4.18
Second-order Entropy (H2): 3.81
Third-order Entropy (H3) of 6-grams: 1.76
Redundancy: 62.52%
```

### Linear B Corpus
```
Linear B Corpus
Vocab Count: 2,426
Grapheme Inventory: 86
Zero-order Entropy (H0): 6.43
First-order Entropy (H1): 5.74
Second-order Entropy (H2): 4.02
Third-order Entropy (H3) of 6-grams: 2.34
Redundancy: 63.61%
```

## References

- Shannon, C. E. (1951). Prediction and Entropy of Printed English. Bell System Technical Journal.
- Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.
- [KenLM Documentation](https://kheafield.com/code/kenlm/)
- https://linearb.xyz/

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to Claude E. Shannon for his groundbreaking work in information theory, which serves as the foundation for this project. Additionally, I extend a heartfelt thank you to Alice Kober for her meticulous and pioneering work deciphering the Linear B script. I stand on the shoulders of giants.