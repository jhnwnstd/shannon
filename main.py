"""Entry point for Shannon entropy analysis.

Usage:
    python main.py english [corpus ...]       English NLTK corpora
    python main.py crosslang [corpus ...]     Linear B + Brown + Europarl
    python main.py analyze [corpus/lang ...]  Full analysis with digrams/trigrams

Examples:
    python main.py english brown reuters
    python main.py crosslang linear_b french
    python main.py analyze brown en de
"""

import logging
import sys

MODES = {"english", "crosslang", "analyze"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    mode = sys.argv[1]
    args = sys.argv[2:]

    if mode not in MODES:
        print(f"Unknown mode: {mode}\n")
        print(__doc__)
        sys.exit(1)

    if mode == "english":
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        from english import run

        run(args or None)

    elif mode == "crosslang":
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        from crosslang import run

        run(args or None)

    elif mode == "analyze":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        from analyzer import run

        run(args or None)


if __name__ == "__main__":
    main()
