# Shannon Entropy Analysis

Character-level entropy and redundancy estimation for natural languages, extending Shannon's 1951 "Prediction and Entropy of Printed English" methodology. Uses KenLM 8-gram models to measure how predictable text is at the letter level.

## Setup

Requires Python 3.11+ and KenLM built from source with 8-gram support.

```bash
# 1. Build KenLM with max order 10 (default 6 is too low for 8-grams)
git clone https://github.com/kpu/kenlm.git
cd kenlm

# Python bindings
MAX_ORDER=10 python setup.py install

# CLI tools (lmplz, build_binary)
mkdir build && cd build
cmake .. -DKENLM_MAX_ORDER=10
make -j4

# Make sure lmplz and build_binary are on PATH
export PATH="$(pwd)/bin:$PATH"

# 2. Install remaining dependencies and NLTK corpora
cd /path/to/this/repo
pip install -r requirements.txt
python -c "import nltk; nltk.download(['brown', 'reuters', 'webtext', 'inaugural', 'nps_chat', 'state_union', 'gutenberg', 'europarl_raw'])"
```

## Scripts

**`english.py`** -- entropy of 7 English NLTK corpora:

```bash
python english.py                # all corpora
python english.py brown reuters  # specific ones
```

**`crosslang.py`** -- Linear B (from `linearb_lexicon.csv`) + Brown + 7 Europarl languages:

```bash
python crosslang.py                    # all
python crosslang.py linear_b french    # specific ones
```

**`analyzer.py`** -- comprehensive analysis with per-file statistics, digram/trigram distributions, transition matrices, and Markov efficiency. All English corpora + 11 Europarl languages:

```bash
python analyzer.py                # all 18 corpus/language combinations
python analyzer.py brown en de    # Brown + Europarl English & German
```

## How it works

Each word (>= 3 chars) is decomposed into space-separated characters so KenLM treats letters as tokens. Four entropy levels are computed:

| Measure | Definition |
|---------|------------|
| H0 | `log2(alphabet_size)` -- maximum possible entropy |
| H1 | Unigram character frequency entropy |
| H2 | Renyi second-order entropy / conditional bigram entropy |
| H3 | KenLM 8-gram model entropy |

Redundancy = `(1 - H3/H0) * 100%`

## Results

### Cross-language comparison

| Language | Alphabet | H0 | H1 | H2 | H3 | Redundancy |
|----------|----------|------|------|------|------|------------|
| Linear B | 86 | 6.43 | 5.74 | 5.46 | 2.34 | 63.54% |
| English | 26 | 4.70 | 4.14 | 3.89 | 1.60 | 65.94% |
| French | 39 | 5.29 | 4.13 | 3.85 | 1.63 | 69.08% |
| German | 30 | 4.91 | 4.17 | 3.78 | 1.39 | 71.68% |
| Italian | 35 | 5.13 | 4.02 | 3.76 | 1.62 | 68.46% |
| Greek | 24 | 4.58 | 4.16 | 3.96 | 1.80 | 60.64% |
| Spanish | 33 | 5.04 | 4.14 | 3.85 | 1.64 | 67.45% |
| Dutch | 28 | 4.81 | 4.09 | 3.70 | 1.40 | 70.82% |
| Portuguese | 39 | 5.24 | 4.19 | 3.09 | 1.63 | 68.91% |
| Swedish | 29 | 4.85 | 4.24 | 3.20 | 1.48 | 69.63% |
| Danish | 27 | 4.85 | 4.14 | 3.17 | 1.41 | 70.97% |
| Finnish | 29 | 5.10 | 4.00 | 3.26 | 1.38 | 75.09% |

### English corpus comparison

| Corpus | Tokens | Vocab | H0 | H1 | H2 | H3 | Redundancy |
|--------|--------|-------|------|------|------|------|------------|
| Brown | 4,369,721 | 46,018 | 4.70 | 4.18 | 3.93 | 1.63 | 65.39% |
| Reuters | 5,845,812 | 28,835 | 4.75 | 4.19 | 3.95 | 1.80 | 62.08% |
| Webtext | 1,193,886 | 16,303 | 5.13 | 4.27 | 4.06 | 1.72 | 66.50% |
| Inaugural | 593,092 | 9,155 | 4.75 | 4.15 | 3.88 | 1.63 | 65.81% |
| State Union | 1,524,983 | 12,233 | 4.81 | 4.16 | 3.91 | 1.67 | 65.17% |
| Gutenberg | 8,123,136 | 41,350 | 4.91 | 4.16 | 3.91 | 1.83 | 62.70% |

### Key findings

- **Germanic languages are most redundant** (DE 71.7%, NL 70.8%, DA 71.0%) -- Romance languages cluster at 65-69%, Greek is lowest at 60.6%
- **Linear B's redundancy (63.5%) is comparable to modern alphabetic languages** despite its 86-character syllabary, suggesting information encoding efficiency is independent of writing system type
- **Finnish is an outlier at 75.1% redundancy**, likely driven by its agglutinative morphology creating highly predictable character sequences
- **English entropy is stable across corpora** (62-66% redundancy), validating the methodology's reliability
- **Universal entropy reduction pattern**: H0 > H1 > H2 > H3 holds for every language tested

## References

- Shannon, C. E. (1951). *Prediction and Entropy of Printed English*. Bell System Technical Journal. (included as `shannon1951.pdf`)
- Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal.
- [KenLM](https://kheafield.com/code/kenlm/)
- [Linear B Lexicon](https://linearb.xyz/)
