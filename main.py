import argparse
from utils import loadModelCLI
from autocomplete import suggestTopKWithBackoff


def run_once(
    input_str,
    vocab,
    ngramCountsList,
    vocab_size,
    topn=3,
    alpha=1e-5,
    startWith=None,
):
    tokens = input_str.strip().lower().split()

    suggestions = suggestTopKWithBackoff(
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

    if not suggestions:
        print("No suggestions found.")
    else:
        print("\nTop Suggestions:")
        for i, (word, prob) in enumerate(suggestions, 1):
            print(f"  {i}. {word} (P={prob:.4f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Akan N-gram Autocomplete CLI")
    parser.add_argument("-i", "--input", type=str, help="Input text (e.g. 'me pɛ')")
    parser.add_argument("-s", "--start", type=str, help="Prefix filter (e.g. 'a')")
    parser.add_argument(
        "-k", "--smoothing", type=float, default=1e-5, help="Smoothing factor"
    )
    parser.add_argument("--topn", type=int, default=3, help="Top-N suggestions")
    args = parser.parse_args()

    vocab, ngramCountsList, vocab_size = loadModelCLI()

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
        # Interactive mode
        print("Akan Autocomplete Shell (type 'exit' to quit)")
        while True:
            user_input = input("Enter input: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting...")
                break

            run_once(
                input_str=user_input,
                vocab=vocab,
                ngramCountsList=ngramCountsList,
                vocab_size=vocab_size,
                topn=args.topn,
                alpha=args.smoothing,
                startWith=args.start,
            )


if __name__ == "__main__":
    main()
