from ngram_model import estimateProbs, getBackOffProb


def suggestWord(
    previousTokens,
    nGramCounts,
    nPlus1GramCounts,
    vocabulary,
    endToken="<e>",
    unknownToken="<unk>",
    k=1.0,
    startWith=None,
):
    n = len(list(nGramCounts.keys())[0])
    previousTokens = ["<s>"] * (n - 1) + previousTokens
    previousNGram = previousTokens[-n:]
    probabilities = estimateProbs(
        previousNGram,
        nGramCounts,
        nPlus1GramCounts,
        vocabulary,
        k=k,
        endToken=endToken,
        unknownToken=unknownToken,
    )
    suggestion = None
    maxProb = 0

    for word, prob in probabilities.items():
        if word in ("<unk>", "<e>"):
            continue
        if startWith is not None and not word.startswith(startWith):
            continue
        if prob > maxProb:
            suggestion = word
            maxProb = prob

    return suggestion, maxProb


def get_suggestions(previousTokens, nGramCountsList, vocabulary, k=1.0, startWith=None):
    modelCounts = len(nGramCountsList)
    suggestions = []
    for i in range(modelCounts - 1):
        nGramCounts = nGramCountsList[i]
        nPlus1GramCounts = nGramCountsList[i + 1]

        suggestion = suggestWord(
            previousTokens,
            nGramCounts,
            nPlus1GramCounts,
            vocabulary,
            k=k,
            startWith=startWith,
        )
        suggestions.append(suggestion)
    return suggestions


def suggestWordWithBackoff(
    previousTokens,
    nGramCountsList,
    vocabulary,
    vocab_size,
    endToken="<e>",
    unknownToken="<unk>",
    alpha=1e-5,
    startWith=None,
):
    n = len(nGramCountsList)
    previousTokens = ["<s>"] * (n - 1) + previousTokens
    previousNGram = previousTokens[-(n - 1) :]

    suggestion = None
    maxProb = 0.0

    for word in vocabulary:
        if word in (unknownToken, endToken):
            continue
        if startWith is not None and not word.startswith(startWith):
            continue

        prob = getBackOffProb(
            word,
            previousNGram,
            nGramCountsList[2],
            nGramCountsList[1],
            nGramCountsList[0],
            vocab_size,
            alpha=alpha,
        )

        if prob > maxProb:
            suggestion = word
            maxProb = prob

    return suggestion, maxProb


def suggestTopKWithBackoff(
    previousTokens,
    unigramCounts,
    bigramCounts,
    trigramCounts,
    vocabulary,
    vocabSize,
    alpha=1e-5,
    topn=3,
    startWith=None,
    unknownToken="<unk>",
    endToken="<e>",
):
    previousTokens = ["<s>", "<s>"] + previousTokens
    context = previousTokens[-2:]
    suggestions = []
    for word in vocabulary:
        if word in (unknownToken, endToken):
            continue
        if startWith and not word.startswith(startWith):
            continue

        prob = getBackOffProb(
            word=word,
            previousNGram=context,
            trigramCounts=trigramCounts,
            bigramCounts=bigramCounts,
            unigramCounts=unigramCounts,
            vocabSize=vocabSize,
            alpha=alpha,
        )
        suggestions.append((word, prob))

    sorted_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
    return sorted_suggestions[:topn]
