import nltk
import torch
import numpy as np
from nltk.util import ngrams
from collections import Counter

class NGgramFeatures():
    def __init__(self, lower, higher = None, featureSpace = []):
        self.nGramCounts = {} #dict of all n-grams in the data and their counts
        self.nGramData = [] #2d list of data (each datapoint has its own list of all n-grams it contains)
        self.allTargets = [] #list of target values (1 or -1) corresponding to the n-gram data
        self.sortedFeatures = featureSpace #list of all the features sorted by their count
        self.allFeatures = [] #list of the filtered features based on the most important features
        self.lower = lower
        self.higher = higher #bounds to create features from n-grams of n between lower and higher
        if higher == None:
            self.higher = lower
        if higher < lower:
            raise ValueError("higher bound cannot be lower than the lower bound")

    #returns n-grams based on the lower and upper bounds
    def nGrams(self, data):
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

    #sort features based on the most frequent n-grams
    def sortFeatures(self):
        self.sortedFeatures = sorted(self.nGramCounts, key=self.nGramCounts.get, reverse=True)

    #truncate the feature space to a fixed length
    def truncateFixed(self, fixedLength):
        self.sortFeatures()
        self.sortedFeatures = self.sortedFeatures[:fixedLength]
        self.truncateData()

    #truncate the feature space to a given fraction of the original length
    def truncatePercentage(self, fraction):
        self.sortFeatures()
        self.sortedFeatures = self.sortedFeatures[:round(len(self.sortedFeatures)*fraction)]
        self.truncateData()

    #truncate the feature space to only include features with a count above the given minimumCount
    def truncateMinimumCount(self, minimumCount):
        self.sortFeatures()
        for i, feature in enumerate(self.sortedFeatures):
            if self.nGramCounts[feature] < minimumCount:
                self.sortedFeatures = self.sortedFeatures[:i]
                break
        
        self.truncateData()

    #truncate the feature space to only include features with a count above the given minimumFraction out of the total count
    def truncateMinimumFraction(self, minimumFraction):
        self.sortFeatures()
        for i, feature in enumerate(self.sortedFeatures):
            if self.nGramCounts[feature] < len(self.sortedFeatures)*minimumFraction:
                self.sortedFeatures = self.sortedFeatures[:i]
                break
        self.truncateData()
    
    #create a feature space based on the most important features chosen by one of the truncate functions
    def truncateData(self):
        for data in self.nGramData:
            c = Counter(data)
            features = []
            for feature in self.sortedFeatures:
                features.append(c[feature])
            self.allFeatures.append(features)
    
    #extract features from a single datapoint (raw text)
    def extractFeatures(self, dataPoint):
        c = Counter(self.nGrams(dataPoint))
        features = []
        for feature in self.sortedFeatures:
            features.append(c[feature])
        return features

    def extractFeatureSpace(self):
        return self.sortedFeatures

    def getTensorFeatures(self):
        return torch.Tensor(self.allFeatures)
    
    def getTensorTargets(self):
        return torch.Tensor(self.allTargets)

    def getFeatures(self):
        return np.array(self.allFeatures)
    
    def getTargets(self):
        return np.array(self.allTargets)