import pickle
import json


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


def loadPickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def loadJSON(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def loadModel(model_dir="models/"):
    vocab = loadJSON(model_dir + "vocabulary.json")
    unigram = loadPickle(model_dir + "unigram.pkl")
    bigram = loadPickle(model_dir + "bigram.pkl")
    trigram = loadPickle(model_dir + "trigram.pkl")
    meta = loadJSON(model_dir + "metadata.json")
    vocab_size = meta["vocab_size"]
    return vocab, unigram, bigram, trigram, vocab_size
