import os
import CustomDataset
from FeatureExtractor import NGgramFeatures

rootdir = '../Spam email database'



def readFile(dir, filename):
    with open(dir + '/' + filename, 'r', encoding='latin1') as file:
        data = file.read()
    return data

featureExtractor = NGgramFeatures(1,2)

#read files and add the data to the feature extractor
for subdir, _, files in os.walk(rootdir):
    if subdir.split('\\')[-1] == 'ham':
        for file in files:
            text = readFile(subdir, file)
            featureExtractor.addDataPoint(text, 1)
    elif subdir.split('\\')[-1] == 'spam':
        for file in files:
            text = readFile(subdir, file)
            featureExtractor.addDataPoint(text, -1)

#featureExtractor.truncateFixed(10)
#print(featureExtractor.getFeatureSpace()[10])

