import os
from FeatureExtractor import NGgramFeatures
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import pickle #save and load models
import random
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler


print('Started reading the data into the feature extractor')

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
featureExtractor.truncateFixed(1000)
features = featureExtractor.getFeatures()
targets = featureExtractor.getTargets()

#save the feature space to a file
with open("featureSpace.txt", 'w') as f:
    featureSpace = featureExtractor.extractFeatureSpace()
    for feature in featureSpace:
        f.write(feature + '\n')

#shuffling the data
zipped = list(zip(features, targets))
random.shuffle(zipped)
features, targets = zip(*zipped)


print('Truncating and shuffling completed, starting learning')

#split into train and test data
test_data_size = 5000
trFeatures = features[:-test_data_size]
trTargets = targets[:-test_data_size]
testFeatures = features[-test_data_size:]
testTargets = targets[-test_data_size:]


'''
#{'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
#https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
#defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(svm.SVC(class_weight = 'balanced'), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(trFeatures, trTargets)

# print best parameter after tuning
print(grid.best_params_)
'''

clf = svm.SVC(C = 100, gamma = 0.001, kernel = 'rbf', class_weight = 'balanced')
clf.fit(trFeatures, trTargets)


print('Finished learning, starting testing')

predictions = clf.predict(testFeatures)
correct = sum(predictions == testTargets)

CM = confusion_matrix(testTargets, predictions)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

print('Finished testing, acc = ' + str(correct/test_data_size))
print('TN:', TN)
print('FN:', FN)
print('TP:', TP)
print('FP:', FP)

#save model
pickle.dump(clf, open("svm.pickle", "wb"))