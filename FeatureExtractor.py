import nltk
import torch
from nltk.util import ngrams

class NGgramFeatures():
    def __init__(self, lower, higher = None):
        self.nGramCounts = {}
        self.nGramData = []
        self.allFeatures = []
        self.allTargets = []
        self.sortedFeatures = []
        self.lower = lower
        if higher == None:
            self.higher = lower
        if higher < lower:
            raise ValueError("higher bound cannot be lower than the lower bound")

    def nGrams(self, data):
        #returns n-grams based on the lower and upper bounds
        returnList = []
        for i in range(self.lower, self.higher + 1):
            nGrams = ngrams(nltk.word_tokenize(data), i)
            returnList += [' '.join(grams) for grams in nGrams]
        return returnList


    def addDataPoint(self, data, target):
        if target == 'ham':
            target = 1
        elif target == 'spam':
            target = -1
        self.allTargets.append(target)
        nGrams = self.nGrams(data)
        self.nGramData.append(nGrams)
        for nGram in nGrams:
            if nGram in self.nGramCounts:
                count = self.nGramCounts[nGram]
                self.nGramCounts.update({nGram: count})
            else:
                self.nGramCounts[nGram] = 1


    def sortFeatures(self):
        self.sortedFeatures = sorted(self.nGramCounts, reverse=True)

    def truncateFixed(self, fixedLength):
        #truncate the feature space to a fixed length
        self.sortFeatures()
        self.sortedFeatures[:fixedLength]
        self.truncateData()

    def truncatePercentage(self, fraction):
        #truncate the feature space to a given fraction of the original length
        self.sortFeatures()
        self.sortedFeatures[:len(self.sortedFeatures)*fraction]
        self.truncateData()

    def truncateMinimumCount(self, minimumCount):
        #truncate the feature space to only include features with a count above the given minimumCount
        self.sortFeatures()
        for i, feature in enumerate(self.sortedFeatures):
            if self.nGramCounts[feature] < minimumCount:
                self.sortedFeatures[:i]
                break
        self.truncateData()

    def truncateMinimumFraction(self, minimumFraction):
        #truncate the feature space to only include features with a count above the given minimumFraction out of the total count
        self.sortFeatures()
        for i, feature in enumerate(self.sortedFeatures):
            if self.nGramCounts[feature] < len(self.sortedFeatures)*minimumFraction:
                self.sortedFeatures[:i]
                break
        self.truncateData()
    
    def truncateData(self):
        #create a feature space based on the most important features chosen by one of the truncate functions
        for data in self.nGramData:
            features = []
            for feature in self.sortedFeatures:
                if feature in data:
                    features.append(data[feature])
                else:
                    features.append(0)
            self.allFeatures.append(features)

    def getFeatureSpace(self):
        return torch.Tensor(self.allFeatures, dtype=torch.int)
    
    def getTargets(self):
        return torch.Tensor(self.allTargets, dtype=torch.int)
