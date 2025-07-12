import pickle
import json
import os
from ngram_model import preprocess, countNGrams


# -- Load Data -- #
def loadData(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# -- Save Utils -- #
def savePickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def saveJson(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -- Train and Save -- #
def train(trainPath, testPath, modelDir="./models/", countThreshold=2):
    os.makedirs(modelDir, exist_ok=True)
    print("1. Loading data from {}...".format(trainPath))
    train = loadData(trainPath)
    test = loadData(testPath)

    print("2. Creating vocabulary and handling OOV...")
    (
        trainData,
        testData,
        vocab,
    ) = preprocess(train, test, countThreshold)

    print("3. Counting N-grams...")
    unigram = countNGrams(trainData, n=1)
    bigram = countNGrams(trainData, n=2)
    trigram = countNGrams(trainData, n=3)

    totalTokens = sum(unigram.values())
    vocabSize = len(vocab)

    print("4. Saving models...")
    saveJson(vocab, os.path.join(modelDir, "vocabulary.json"))
    savePickle(unigram, os.path.join(modelDir, "unigram.pkl"))
    savePickle(bigram, os.path.join(modelDir, "bigram.pkl"))
    savePickle(trigram, os.path.join(modelDir, "trigram.pkl"))

    saveJson(
        {
            "total_tokens": totalTokens,
            "vocab_size": vocabSize,
            "count_threshold": countThreshold,
        },
        os.path.join(modelDir, "metadata.json"),
    )

    print("Model trained and saved to:", modelDir)


if __name__ == "__main__":
    train("./data/processed/trainTokens.pkl", "./data/processed/testTokens.pkl")
