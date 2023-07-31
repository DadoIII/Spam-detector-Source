import pickle
from FeatureExtractor import NGgramFeatures
from sklearn import svm

text = input("Input your email in raw text: ")

featureSpace = []
with open("featureSpace.txt", "r") as f:
    line = f.readline().strip()
    while line != "":
        featureSpace.append(line)
        line = f.readline().strip()

fe = NGgramFeatures(1, 2, featureSpace = featureSpace)

# load model
loaded_svm = pickle.load(open("svm.pickle", "rb"))

# you can use loaded model to compute predictions
y_predicted = loaded_svm.predict([fe.extractFeatures(text)])

if y_predicted == 1:
    print("The email is ham!")
else:
    print("The email is spam!")