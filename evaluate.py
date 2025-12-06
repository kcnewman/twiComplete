import math
from ngram_model import getBackOffProb, replaceOOV
from utils import loadPickle, loadModel


def computePerplexity(
    tokenized_sentences,
    unigram_counts,
    bigram_counts,
    trigram_counts,
    vocab_size,
    alpha=1e-5,
):
    total_log_prob = 0.0
    total_tokens = 0

    for sentence in tokenized_sentences:
        sentence = ["<s>", "<s>"] + sentence + ["<e>"]
        for i in range(2, len(sentence)):
            context = sentence[i - 2 : i]
            target = sentence[i]
            prob = getBackOffProb(
                word=target,
                previousNGram=context,
                trigramCounts=trigram_counts,
                bigramCounts=bigram_counts,
                unigramCounts=unigram_counts,
                vocabSize=vocab_size,
                alpha=alpha,
            )
            total_log_prob += math.log(prob)
            total_tokens += 1

    avg_log_prob = total_log_prob / total_tokens
    perplexity = math.exp(-avg_log_prob)
    return perplexity


if __name__ == "__main__":
    vocab, unigram, bigram, trigram, vocab_size = loadModel("./models/")

    test_data = loadPickle("./data/processed/testTokens.pkl")

    test_data = replaceOOV(test_data, vocabulary=set(vocab), unknownTokens="<unk>")

    ppl = computePerplexity(
        test_data,
        unigram_counts=unigram,
        bigram_counts=bigram,
        trigram_counts=trigram,
        vocab_size=vocab_size,
        alpha=1e-5,
    )

    print(f"Perplexity on test set: {ppl:.2f}")
