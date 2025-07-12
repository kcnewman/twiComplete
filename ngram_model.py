def nplusFreqWords(tokenizedSentences, countTreshold):
    """
    Creat Vocab list by selecting words that appear N times or more to handle OOV Words

    Returns:
        Closed Vocabulary dictionary
    """
    closedVocab = []

    def getWordCounts(tokenizedSentences):
        wordCount = {}
        for sentence in tokenizedSentences:
            for token in sentence:
                if token not in wordCount:
                    wordCount[token] = 1
                else:
                    wordCount[token] += 1
        return wordCount

    wordCounts = getWordCounts(tokenizedSentences)
    for word, count in wordCounts.items():
        if count >= countTreshold:
            closedVocab.append(word)
    return closedVocab


def replaceOOV(tokenizedSentences, vocabulary, unknownTokens="<unk>"):
    vocabulary = set(vocabulary)
    replacedTokens = []
    for sentence in tokenizedSentences:
        replacedSentence = []
        for token in sentence:
            if token in vocabulary:
                replacedSentence.append(token)
            else:
                replacedSentence.append(unknownTokens)
        replacedTokens.append(replacedSentence)
    return replacedTokens


def preprocess(
    trainData,
    testData,
    countTreshold,
    unknownToken="<unk>",
    nplusFreq=nplusFreqWords,
    replaceOOV=replaceOOV,
):
    vocabulary = nplusFreqWords(trainData, countTreshold=countTreshold)
    trainDataReplaced = replaceOOV(trainData, vocabulary, unknownToken)
    testDataReplaced = replaceOOV(testData, vocabulary, unknownToken)

    return trainDataReplaced, testDataReplaced, vocabulary


# --- N-Gram Model ---#
def countNGrams(data, n, startToken="<s>", endToken="<e>"):
    nGrams = {}
    for sentence in data:
        sentence = [startToken] * n + sentence + [endToken]
        sentence = tuple(sentence)
        for i in range(len(sentence) - n + 1):
            nGram = sentence[i : i + n]
            if nGram in nGrams:
                nGrams[nGram] += 1
            else:
                nGrams[nGram] = 1
    return nGrams


def estimateProb(word, previousNGram, nGramCounts, nPlus1GramCounts, vocabSize, k=1.0):
    previousNGram = tuple(previousNGram)
    previousNGramCount = nGramCounts.get(previousNGram, 0)
    denominator = previousNGramCount + k * vocabSize
    nPlus1Gram = previousNGram + (word,)
    nPlus1GramCount = nPlus1GramCounts.get(nPlus1Gram, 0)
    numerator = nPlus1GramCount + k
    probability = numerator / denominator
    return probability


def estimateProbs(
    previousNGram,
    nGramCounts,
    nPlus1GramCounts,
    vocab,
    endToken="<e>",
    unknownToken="<unk>",
    k=1.0,
):
    previousNGram = tuple(previousNGram)
    vocab = vocab + [endToken, unknownToken]
    vocabSize = len(vocab)
    probabilities = {}
    for word in vocab:
        probability = estimateProb(
            word, previousNGram, nGramCounts, nPlus1GramCounts, vocabSize, k=k
        )
        probabilities[word] = probability
    return probabilities


def getBackOffProb(
    word,
    previousNGram,
    trigramCounts,
    bigramCounts,
    unigramCounts,
    vocabSize,
    alpha=1e-5,
):
    """
    Estimate P(word | context) using backoff:
    - Try trigram: P(w | w_{-2}, w_{-1})
    - If not found, try bigram: P(w | w_{-1})
    - If not found, use unigram: P(w)
    - All smoothed using Laplace smoothing or additive smoothing (via alpha)
    """
    word = word.lower()
    previousNGram = list(previousNGram)

    # --- Trigram ---
    if len(previousNGram) >= 2:
        trigram = tuple(previousNGram[-2:] + [word])
        bigram = tuple(previousNGram[-2:])
        trigram_count = trigramCounts.get(trigram, 0)
        bigram_count = bigramCounts.get(bigram, 0)
        if trigram_count > 0 or bigram_count > 0:
            return (trigram_count + alpha) / (bigram_count + alpha * vocabSize)

    # --- Bigram ---
    if len(previousNGram) >= 1:
        bigram = tuple(previousNGram[-1:] + [word])
        unigram = tuple(previousNGram[-1:])
        bigram_count = bigramCounts.get(bigram, 0)
        unigram_count = unigramCounts.get(unigram, 0)
        if bigram_count > 0 or unigram_count > 0:
            return (bigram_count + alpha) / (unigram_count + alpha * vocabSize)

    # --- Unigram ---
    unigram_count = unigramCounts.get((word,), 0)
    total_tokens = sum(unigramCounts.values())
    return (unigram_count + alpha) / (total_tokens + alpha * vocabSize)
