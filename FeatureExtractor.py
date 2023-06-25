import nltk
import torch
import numpy as np
from nltk.util import ngrams
from collections import Counter

class NGgramFeatures():
    def __init__(self, lower, higher = None):
        self.nGramCounts = {}
        self.nGramData = []
        self.allFeatures = []
        self.allTargets = []
        self.sortedFeatures = []
        self.lower = lower
        self.higher = higher
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
                self.nGramCounts.update({nGram: count + 1})
            else:
                self.nGramCounts[nGram] = 1


    def sortFeatures(self):
        self.sortedFeatures = sorted(self.nGramCounts, key=self.nGramCounts.get, reverse=True)

    def truncateFixed(self, fixedLength):
        #truncate the feature space to a fixed length
        self.sortFeatures()
        self.sortedFeatures = self.sortedFeatures[:fixedLength]
        self.truncateData()

    def truncatePercentage(self, fraction):
        #truncate the feature space to a given fraction of the original length
        self.sortFeatures()
        self.sortedFeatures = self.sortedFeatures[:round(len(self.sortedFeatures)*fraction)]
        self.truncateData()

    def truncateMinimumCount(self, minimumCount):
        #truncate the feature space to only include features with a count above the given minimumCount
        self.sortFeatures()
        for i, feature in enumerate(self.sortedFeatures):
            if self.nGramCounts[feature] < minimumCount:
                self.sortedFeatures = self.sortedFeatures[:i]
                break
        
        self.truncateData()

    def truncateMinimumFraction(self, minimumFraction):
        #truncate the feature space to only include features with a count above the given minimumFraction out of the total count
        self.sortFeatures()
        for i, feature in enumerate(self.sortedFeatures):
            if self.nGramCounts[feature] < len(self.sortedFeatures)*minimumFraction:
                self.sortedFeatures = self.sortedFeatures[:i]
                break
        self.truncateData()
    
    def truncateData(self):
        #create a feature space based on the most important features chosen by one of the truncate functions
        for data in self.nGramData:
            c = Counter(data)
            features = []
            for feature in self.sortedFeatures:
                features.append(c[feature])
            self.allFeatures.append(features)

    def getTensorFeatures(self):
        return torch.Tensor(self.allFeatures)
    
    def getTensorTargets(self):
        return torch.Tensor(self.allTargets)

    def getFeatures(self):
        return np.array(self.allFeatures)
    
    def getTargets(self):
        return np.array(self.allTargets)