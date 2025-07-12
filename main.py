#!/usr/bin/env python3
"""
Akan N-gram Autocomplete CLI
Copyright (c) 2025 Kelvin Newman
Date: July 12, 2025
License: MIT

A tool for interactively exploring next-word suggestions based on an N-gram language model trained on Akan text.

Features:
- Word-by-word interactive shell
- Trigram backoff with smoothing
- Top-N ranked predictions
"""

import argparse
import sys
from utils import loadModelCLI
from autocomplete import suggestTopKWithBackoff


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def display_suggestions(tokens, suggestions):
    print(f"\n{bcolors.OKGREEN}Top suggestions for: {' '.join(tokens)}{bcolors.ENDC}")
    for i, (word, prob) in enumerate(suggestions, 1):
        print(f"  {i}. {bcolors.OKBLUE}{word}{bcolors.ENDC} (P={prob:.4f})")
    print()


def suggest_next_word(
    tokens, vocab, ngramCountsList, vocab_size, topn=3, alpha=1e-5, startWith=None
):
    return suggestTopKWithBackoff(
        previousTokens=tokens,
        unigramCounts=ngramCountsList[0],
        bigramCounts=ngramCountsList[1],
        trigramCounts=ngramCountsList[2],
        vocabulary=vocab,
        vocabSize=vocab_size,
        alpha=alpha,
        topn=topn,
        startWith=startWith,
    )


def interactive_shell(vocab, ngramCountsList, vocab_size, topn=3, alpha=1e-5):
    print(f"""{bcolors.BOLD}
==========================================
🌍 Akan N-gram Autocomplete CLI (2025)
Developed by Kelvin Newman
------------------------------------------
Type Akan words one at a time to get suggestions.
Commands:
  - /new    → Start a new sentence
  - exit    → Quit the shell
==========================================
{bcolors.ENDC}""")

    current_input = []

    while True:
        try:
            word = input(f"{bcolors.OKCYAN}Next word > {bcolors.ENDC}").strip().lower()

            if word in {"exit", "quit"}:
                print(
                    f"{bcolors.WARNING}👋 Exiting autocomplete shell...{bcolors.ENDC}"
                )
                break

            elif word == "/new":
                current_input = []
                print(f"{bcolors.OKBLUE}🆕 Starting a new sentence...{bcolors.ENDC}")
                continue

            elif not word:
                continue

            current_input.append(word)

            suggestions = suggest_next_word(
                tokens=current_input,
                vocab=vocab,
                ngramCountsList=ngramCountsList,
                vocab_size=vocab_size,
                topn=topn,
                alpha=alpha,
            )

            if suggestions:
                display_suggestions(current_input, suggestions)
            else:
                print(
                    f"{bcolors.FAIL}No suggestions found for: {' '.join(current_input)}{bcolors.ENDC}"
                )

        except KeyboardInterrupt:
            print(f"\n{bcolors.WARNING}👋 Exiting autocomplete shell...{bcolors.ENDC}")
            break
        except Exception as e:
            print(f"{bcolors.FAIL}Error: {e}{bcolors.ENDC}", file=sys.stderr)


def run_once(
    input_str, vocab, ngramCountsList, vocab_size, topn=3, alpha=1e-5, startWith=None
):
    tokens = input_str.strip().lower().split()
    suggestions = suggest_next_word(
        tokens,
        vocab,
        ngramCountsList,
        vocab_size,
        topn,
        alpha,
        startWith,
    )

    if suggestions:
        display_suggestions(tokens, suggestions)
    else:
        print(f"{bcolors.FAIL}No suggestions found.{bcolors.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description="🌍 Akan N-gram Autocomplete CLI (Kelvin Newman, 2025)"
    )
    parser.add_argument("-i", "--input", type=str, help="Input text (e.g. 'me pɛ')")
    parser.add_argument("-s", "--start", type=str, help="Prefix filter (e.g. 'a')")
    parser.add_argument(
        "-k", "--smoothing", type=float, default=1e-5, help="Smoothing factor"
    )
    parser.add_argument("--topn", type=int, default=3, help="Top-N suggestions to show")

    args = parser.parse_args()

    print("🚀 Loading model...")
    vocab, ngramCountsList, vocab_size = loadModelCLI()
    print(f"✅ Model loaded. Vocabulary size: {vocab_size}")

    if args.input:
        run_once(
            input_str=args.input,
            vocab=vocab,
            ngramCountsList=ngramCountsList,
            vocab_size=vocab_size,
            topn=args.topn,
            alpha=args.smoothing,
            startWith=args.start,
        )
    else:
        interactive_shell(
            vocab=vocab,
            ngramCountsList=ngramCountsList,
            vocab_size=vocab_size,
            topn=args.topn,
            alpha=args.smoothing,
        )


if __name__ == "__main__":
    main()
