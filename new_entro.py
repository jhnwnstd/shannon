import numpy as np
from collections import Counter, defaultdict
import regex as re
from typing import Dict, List, Optional, Tuple
import nltk
from nltk.corpus import *
import logging
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class EntropyResults:
    """Container for entropy calculation results"""
    h0: float  # Maximum possible entropy
    h1: float  # First-order entropy
    h2: float  # Second-order entropy
    h3: float  # Third-order entropy
    alphabet_size: int
    unique_digrams: int
    unique_trigrams: int
    total_chars: int
    char_distribution: Dict[str, float]
    digram_distribution: Dict[str, Dict[str, float]]
    trigram_distribution: Dict[str, Dict[str, float]]
    transitions: Dict[str, Dict[str, float]]
    markov_efficiency: float
    compression_ratio: float
    predictability: float
    branching_factor: float

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

class ShannonAnalyzer:
    def __init__(self, ngram_order: int = 8):
        self.ngram_order = ngram_order
        self._setup_logging()
        self._download_corpora()
        
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _download_corpora(self) -> None:
        required = ['gutenberg', 'brown', 'reuters']
        for corpus in required:
            try:
                nltk.data.find(f'corpora/{corpus}')
            except LookupError:
                self.logger.info(f"Downloading {corpus} corpus...")
                nltk.download(corpus)

    def preprocess_text(self, text: str) -> str:
        # Normalize text with sentence boundaries
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return f"^ {text.lower().strip()} $"

    def calculate_ngram_stats(self, text: str, n: int) -> Tuple[Counter, int]:
        ngrams = [''.join(gram) for gram in zip(*[text[i:] for i in range(n)])]
        return Counter(ngrams), len(ngrams)

    def calculate_entropy(self, freq: Counter, total: int, 
                        prev_freq: Optional[Counter] = None,
                        prev_total: Optional[int] = None) -> float:
        if prev_freq is None:
            # Zero-order or first-order entropy
            probs = [count/total for count in freq.values()]
            return -sum(p * np.log2(p) for p in probs)
        
        # Conditional entropy
        entropy = 0
        for seq, count in freq.items():
            prefix = seq[:-1]
            p_seq = count / total
            p_prev = prev_freq[prefix] / prev_total if prefix in prev_freq else 0
            
            if p_prev > 0:
                entropy -= p_seq * np.log2(p_seq / p_prev)
        return entropy

    def analyze_text(self, text: str) -> EntropyResults:
        if not text or len(text) < self.ngram_order:
            raise ValueError(f"Text too short, minimum length: {self.ngram_order}")
        
        processed_text = self.preprocess_text(text)
        
        # Calculate n-gram statistics
        char_freq, total_chars = self.calculate_ngram_stats(processed_text, 1)
        digram_freq, total_digrams = self.calculate_ngram_stats(processed_text, 2)
        trigram_freq, total_trigrams = self.calculate_ngram_stats(processed_text, 3)
        
        # Calculate entropies
        h0 = np.log2(len(char_freq))
        h1 = self.calculate_entropy(char_freq, total_chars)
        h2 = self.calculate_entropy(digram_freq, total_digrams, char_freq, total_chars)
        h3 = self.calculate_entropy(trigram_freq, total_trigrams, digram_freq, total_digrams)
        
        # Calculate distributions
        char_dist = {c: count/total_chars for c, count in char_freq.most_common(10)}
        
        # Calculate transition matrices
        transitions = defaultdict(lambda: defaultdict(float))
        for digram, count in digram_freq.items():
            first, second = digram
            transitions[first][second] = count / char_freq[first]
        
        # Calculate advanced metrics
        markov_efficiency = 100 * (h1 - h2) / h1
        compression_ratio = h3 / h0
        predictability = 100 * (1 - h3/h0)
        branching_factor = np.mean([len(trans) for trans in transitions.values()])
        
        return EntropyResults(
            h0=h0,
            h1=h1,
            h2=h2,
            h3=h3,
            alphabet_size=len(char_freq),
            unique_digrams=len(digram_freq),
            unique_trigrams=len(trigram_freq),
            total_chars=total_chars,
            char_distribution=char_dist,
            digram_distribution=dict(transitions),
            trigram_distribution={},  # Simplified for now
            transitions=dict(transitions),
            markov_efficiency=markov_efficiency,
            compression_ratio=compression_ratio,
            predictability=predictability,
            branching_factor=branching_factor
        )

    def analyze_corpus(self, corpus_name: str, max_files: Optional[int] = None) -> CorpusStatistics:
        self.logger.info(f"Starting analysis of {corpus_name} corpus...")
        
        try:
            corpus = getattr(nltk.corpus, corpus_name)
            file_ids = corpus.fileids()[:max_files] if max_files else corpus.fileids()
            
            results = []
            total_chars = 0
            
            for file_id in tqdm(file_ids, desc="Analyzing files"):
                try:
                    text = corpus.raw(file_id)
                    result = self.analyze_text(text)
                    results.append(result)
                    total_chars += result.total_chars
                except Exception as e:
                    self.logger.warning(f"Error processing {file_id}: {str(e)}")
            
            return self._calculate_corpus_statistics(corpus_name, results, total_chars)
            
        except Exception as e:
            self.logger.error(f"Error analyzing corpus: {str(e)}")
            raise

    def _calculate_corpus_statistics(self, corpus_name: str, 
                                   results: List[EntropyResults], 
                                   total_chars: int) -> CorpusStatistics:
        if not results:
            return CorpusStatistics(
                corpus_name=corpus_name,
                files_analyzed=0,
                total_chars=0,
                mean_entropy={},
                std_entropy={},
                reductions={},
                patterns={},
                efficiency_metrics={}
            )

        # Calculate mean and std for entropy measures
        entropy_metrics = ['h0', 'h1', 'h2', 'h3']
        mean_entropy = {
            metric: np.mean([getattr(r, metric) for r in results])
            for metric in entropy_metrics
        }
        std_entropy = {
            metric: np.std([getattr(r, metric) for r in results])
            for metric in entropy_metrics
        }

        # Calculate reductions
        reductions = {
            'h0_to_h1': 100 * (1 - mean_entropy['h1'] / mean_entropy['h0']),
            'h1_to_h2': 100 * (1 - mean_entropy['h2'] / mean_entropy['h1']),
            'h2_to_h3': 100 * (1 - mean_entropy['h3'] / mean_entropy['h2']),
            'total': 100 * (1 - mean_entropy['h3'] / mean_entropy['h0'])
        }

        # Calculate efficiency metrics
        efficiency_metrics = {
            'markov_efficiency': np.mean([r.markov_efficiency for r in results]),
            'compression_ratio': np.mean([r.compression_ratio for r in results]),
            'predictability': np.mean([r.predictability for r in results]),
            'branching_factor': np.mean([r.branching_factor for r in results])
        }

        # Aggregate patterns
        char_patterns = defaultdict(float)
        for r in results:
            for char, freq in r.char_distribution.items():
                char_patterns[char] += freq

        patterns = {
            'chars': dict(sorted(char_patterns.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:5])
        }

        return CorpusStatistics(
            corpus_name=corpus_name,
            files_analyzed=len(results),
            total_chars=total_chars,
            mean_entropy=mean_entropy,
            std_entropy=std_entropy,
            reductions=reductions,
            patterns=patterns,
            efficiency_metrics=efficiency_metrics
        )

def print_analysis(stats: CorpusStatistics) -> None:
    print(f"\n{stats.corpus_name} Corpus Analysis")
    print("=" * 50)
    print(f"Files analyzed: {stats.files_analyzed}")
    print(f"Total characters: {stats.total_chars:,}")
    
    print("\nEntropy Measures (bits)")
    print("-" * 30)
    for order in ['h0', 'h1', 'h2', 'h3']:
        print(f"{order.upper()}: {stats.mean_entropy[order]:.2f} ± "
              f"{stats.std_entropy[order]:.2f}")
    
    print("\nInformation Reduction")
    print("-" * 30)
    print(f"H0 → H1: {stats.reductions['h0_to_h1']:.1f}%")
    print(f"H1 → H2: {stats.reductions['h1_to_h2']:.1f}%")
    print(f"H2 → H3: {stats.reductions['h2_to_h3']:.1f}%")
    print(f"Total:   {stats.reductions['total']:.1f}%")
    
    print("\nEfficiency Metrics")
    print("-" * 30)
    print(f"Markov efficiency: {stats.efficiency_metrics['markov_efficiency']:.1f}%")
    print(f"Compression ratio: {stats.efficiency_metrics['compression_ratio']:.2f}")
    print(f"Predictability:    {stats.efficiency_metrics['predictability']:.1f}%")
    print(f"Branching factor:  {stats.efficiency_metrics['branching_factor']:.1f}")
    
    print("\nMost Common Characters")
    print("-" * 30)
    for char, freq in stats.patterns['chars'].items():
        print(f"'{char}': {freq*100:.1f}%")

if __name__ == "__main__":
    analyzer = ShannonAnalyzer()
    results = analyzer.analyze_corpus('gutenberg', max_files=10)
    print_analysis(results)