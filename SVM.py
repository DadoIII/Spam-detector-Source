import os
import CustomDataset
from FeatureExtractor import NGgramFeatures
from sklearn import svm
from sklearn.model_selection import GridSearchCV
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
featureExtractor.truncateFixed(200)
features = featureExtractor.getFeatures()
targets = featureExtractor.getTargets()

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

#{'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
#https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(svm.SVC(class_weight = 'balanced'), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(trFeatures, trTargets)

# print best parameter after tuning
print(grid.best_params_)

'''
clf = svm.SVC(C = 0.8, class_weight = 'balanced')
#clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(trFeatures, trTargets)


print('Finished learning, starting testing')

predictions = clf.predict(testFeatures)
correct = sum(predictions == testTargets)


print('Finished testing, acc = ' + str(correct/test_data_size))
'''
