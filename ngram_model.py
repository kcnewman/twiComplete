import pickle


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
    trainDataReplaced = replaceOOV(
        trainData, unknownToken=unknownToken, vocabulary=vocabulary
    )
    testDataReplaced = replaceOOV(
        trainData, unknownToken=unknownToken, vocabulary=vocabulary
    )

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
