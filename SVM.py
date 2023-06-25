import os
import CustomDataset
from FeatureExtractor import NGgramFeatures
from sklearn import svm
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler

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

print('Features extracted, starting truncating')
featureExtractor.truncateFixed(10)
features = featureExtractor.getFeatures()
targets = featureExtractor.getTargets()
print('Truncating completed, starting learning')

trFeatures = features[:-100]
trTargets = targets[:-100]
testFeatures = features[-100:]
testTargets = targets[-100:]

clf = svm.SVC()
#clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(trFeatures, trTargets)

print('Finished learning, starting testing')

predictions = clf.predict(testFeatures)
acc = sum(predictions == testTargets)

print('Finished testing, acc = ' + str(acc/100))

