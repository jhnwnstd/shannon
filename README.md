# Shannon Entropy Analysis Project

## Overview

This project implements a computational method for estimating language word-level entropy and redundancy, inspired by Claude E. Shannon's paper, "Prediction and Entropy of Printed English" (1951). While Shannon's original work focused only on English, this project extends the methodology to analyze:
- Multiple English corpora (Brown, Reuters, Webtext, etc.)
- European languages from the Europarl corpus
- The Linear B script (writing system for Mycenaean Greek)

## Background

In his 1951 paper, Shannon introduced a method to estimate a language's entropy and redundancy by leveraging the predictability of characters based on preceding text. Entropy measures the average information content per symbol, while redundancy indicates the degree to which a language constrains or structures its text, making some sequences more predictable.

Shannon's experiments involved predicting the next letter in a sequence of text, and he used this predictability to calculate entropy. Higher predictability means lower entropy because the next letter can be guessed with higher accuracy, resulting in less new information.

This project replicates and extends Shannon's methodology using KenLM. This modern language modeling tool builds n-gram models to predict the next character in a sequence based on preceding characters.

## Project Structure

- `entropy_model/`: Directory for trained KenLM models
- `Linear_B_Lexicon.csv`: Input corpus file containing Linear B words
- `linearb_entro.py`: Script for Linear B and European language analysis
- `new_entro.py`: Script for detailed entropy analysis
- `shannon_entro.py`: Script for English corpora analysis
- `Prediction_and_Entropy_of_Printed_English.pdf`: Shannon's original paper
- `README.md`: This document

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
 ```bash
   pip install https://github.com/kpu/kenlm/archive/master.zip
 ```

3. **Download NLTK Data**:
 ```python
   import nltk
   nltk.download(['brown', 'reuters', 'webtext', 'inaugural', 
                 'nps_chat', 'state_union', 'gutenberg', 'europarl_raw'])
 ```

4. **Prepare Corpus**:
 Place `Linear_B_Lexicon.csv` in the project directory.

## Usage

1. **Analyze English Corpora**:
 ```bash
   python shannon_entro.py
 ```

2. **Analyze Linear B and European Languages**:
 ```bash
   python linearb_entro.py
 ```

## Methodology

1. **Load and Format Corpus**:
   - Clean text using language-specific character filters
   - Handle special characters and diacritics
   - Support Unicode ranges for Linear B (U+10000 to U+100FF)
   - Remove duplicates and non-character content

2. **Build KenLM Model**:
   - Create 8-gram language models
   - Process text as character sequences
   - Generate both ARPA and binary model formats

3. **Calculate Entropy**:
   - **H0 (Zero-order)**: `log2(alphabet_size)`
   - **H1 (First-order)**: Unigram frequency-based entropy
   - **H2 (Second-order/Rényi)**: `-log2(sum(probabilities²))`
   - **H3 (Third-order)**: Using KenLM 8-gram predictions

4. **Calculate Redundancy**:
 ```python
   redundancy = (1 - H3 / H0) * 100
 ```

## Findings

### European Languages Analysis

| Language | Grapheme Inventory | H0 | H1 | H2 | H3 | Redundancy |
|----------|-------------------|-----|-----|-----|-----|------------|
| Linear B | 86 | 6.43 | 5.74 | 5.46 | 2.34 | 63.54% |
| English (Europarl) | 26 | 4.70 | 4.14 | 3.89 | 1.60 | 65.94% |
| French | 39 | 5.29 | 4.13 | 3.85 | 1.63 | 69.08% |
| German | 30 | 4.91 | 4.17 | 3.78 | 1.39 | 71.68% |
| Italian | 35 | 5.13 | 4.02 | 3.76 | 1.62 | 68.46% |
| Greek | 24 | 4.58 | 4.16 | 3.96 | 1.80 | 60.64% |
| Spanish | 33 | 5.04 | 4.14 | 3.85 | 1.64 | 67.45% |
| Dutch | 28 | 4.81 | 4.09 | 3.70 | 1.40 | 70.82% |

### English Corpus Comparison

| Corpus | Token Count | Vocab Count | H0 | H1 | H2 | H3 | Redundancy |
|--------|-------------|-------------|-----|-----|-----|-----|------------|
| Brown | 4,369,721 | 46,018 | 4.70 | 4.18 | 3.93 | 1.63 | 65.39% |
| Reuters | 5,845,812 | 28,835 | 4.75 | 4.19 | 3.95 | 1.80 | 62.08% |
| Webtext | 1,193,886 | 16,303 | 5.13 | 4.27 | 4.06 | 1.72 | 66.50% |
| Inaugural | 593,092 | 9,155 | 4.75 | 4.15 | 3.88 | 1.63 | 65.81% |
| State Union | 1,524,983 | 12,233 | 4.81 | 4.16 | 3.91 | 1.67 | 65.17% |
| Gutenberg | 8,123,136 | 41,350 | 4.91 | 4.16 | 3.91 | 1.83 | 62.70% |

### Key Findings

1. **Writing System Complexity**
   - Linear B's large grapheme inventory (86) yields higher absolute entropy
   - Redundancy remains comparable to modern languages (63.54%)
   - Suggests information encoding efficiency is independent of writing system complexity

2. **Language Family Patterns**
   - Germanic languages show highest redundancy (German: 71.68%, Dutch: 70.82%)
   - Romance languages show moderate redundancy (65-69%)
   - Modern Greek shows the lowest redundancy (60.64%)
   - Suggests systematic differences in information encoding across language families

3. **Corpus Effects**
   - English shows consistent redundancy (62-66%) across different corpora
   - Larger corpora (>5M tokens) tend toward slightly lower redundancy
   - Validates methodology's reliability for cross-linguistic comparison

4. **Information Structure**
   - All languages show consistent entropy reduction pattern (H0 > H1 > H2 > H3)
   - Rate of reduction varies by language family
   - Suggests universal principles in linguistic information structure

## References

- Shannon, C. E. (1951). Prediction and Entropy of Printed English. Bell System Technical Journal.
- Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.
- [KenLM Documentation](https://kheafield.com/code/kenlm/)
- https://linearb.xyz/

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to Claude E. Shannon for his groundbreaking work in information theory and to Alice Kober for her pioneering work deciphering the Linear B script. The project also builds upon the excellent KenLM language modeling toolkit and NLTK's comprehensive corpus collection.