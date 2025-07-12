import argparse
import readline  # For better CLI experience on Unix
from utils import loadModelCLI
from autocomplete import suggestTopKWithBackoff
from datetime import datetime


def print_banner():
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🌟 AKAN AUTOCOMPLETE CLI 🌟                         ║
║                     Intelligent N-gram Language Predictor                    ║
║                           for the Akan Language                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  © {datetime.now().year} TwiComplete Project. Built with ❤️ for Akan NLP.        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def tokenize_input(input_str):
    import re

    return re.findall(r"[a-zA-ZɛɔƐƆŋŊ']+", input_str.lower())


def print_suggestions(suggestions):
    if not suggestions:
        print("\n┌────────────────────────────────────────────┐")
        print("│ 😔 No suggestions found.                  │")
        print("└────────────────────────────────────────────┘")
        return

    print("\n┌────────────────────────────────────────────┐")
    print(f"│ 🎯 Top {len(suggestions)} Suggestions")
    print("├────────────────────────────────────────────┤")
    for i, (word, prob) in enumerate(suggestions, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
        print(f"│ {emoji} {i}. {word:<15} (P={prob:.6f})")
    print("└────────────────────────────────────────────┘")


def run_once(
    input_str, vocab, ngramCountsList, vocab_size, topn=3, alpha=1e-5, startWith=None
):
    tokens = tokenize_input(input_str)
    if not tokens:
        print("❌ No valid tokens in input.")
        return

    print(f"🔍 Context: {' → '.join(tokens)}")

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
    print_suggestions(suggestions)


def run_interactive(vocab, ngramCountsList, vocab_size, topn, alpha):
    print_banner()
    print("📚 Interactive Mode — Start typing! ('exit' to quit)\n")
    context = []

    while True:
        try:
            word = input("📝 Next word: ").strip().lower()
            if word in {"exit", "quit"}:
                print("👋 Goodbye! Thanks for using Akan Autocomplete CLI.")
                break
            elif not word:
                print("⚠️  Please enter a word.")
                continue

            context.append(word)
            print(f"🔍 Context: {' → '.join(context)}")

            suggestions = suggestTopKWithBackoff(
                previousTokens=context,
                unigramCounts=ngramCountsList[0],
                bigramCounts=ngramCountsList[1],
                trigramCounts=ngramCountsList[2],
                vocabulary=vocab,
                vocabSize=vocab_size,
                alpha=alpha,
                topn=topn,
            )
            print_suggestions(suggestions)

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="🌟 Akan N-gram Autocomplete CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-i", "--input", type=str, help="📝 Input text (e.g. 'me pɛ')")
    parser.add_argument("-s", "--start", type=str, help="🎯 Prefix filter (e.g. 'a')")
    parser.add_argument(
        "-k", "--smoothing", type=float, default=1e-5, help="🧮 Smoothing factor"
    )
    parser.add_argument("--topn", type=int, default=3, help="🏆 Top-N suggestions")
    parser.add_argument(
        "--interactive", action="store_true", help="💬 Launch interactive mode"
    )
    args = parser.parse_args()

    print("🚀 Loading model...")
    vocab, ngramCountsList, vocab_size = loadModelCLI()
    print(f"✅ Model loaded. Vocabulary size: {vocab_size}")

    if args.interactive:
        run_interactive(
            vocab, ngramCountsList, vocab_size, topn=args.topn, alpha=args.smoothing
        )
    elif args.input:
        print_banner()
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
        print(
            "ℹ️ No input provided. Use -i to provide input or --interactive to launch interactive mode."
        )


if __name__ == "__main__":
    main()
