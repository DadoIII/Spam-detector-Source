import nltk
from nltk.util import ngrams

class FeatureExtractor():
    def __init__(self):
        pass

    def nGrams(self, data, n):
        nGrams = ngrams(nltk.word_tokenize(data), n)
        return [' '.join(grams) for grams in nGrams]

    def nultipleNGrams(self, data, lower, higher):
        returnList = []
        for i in range(lower, higher + 1):
            returnList += self.nGrams(data, i)
        return returnList