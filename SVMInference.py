import pickle
from FeatureExtractor import NGgramFeatures
from sklearn import svm

#text = input("Input your email in raw text: ")
text = """Subject: objectives session
in the last cao staff meeting rick stated that he wanted to set up a 1 1 / 2
hour session to discuss objectives for 2000 . this session has been set for
tuesday , february 22 from 1 : 30 - 3 : 00 in 49 c 4 .
each of you should come prepared to briefly discuss the goals that have been
set for your group . if you will not be able to attend this meeting please
let me know asap .
"""

#import the feature space
featureSpace = []
with open("featureSpace.txt", "r") as f:
    line = f.readline().strip()
    while line != "":
        featureSpace.append(line)
        line = f.readline().strip()

fe = NGgramFeatures(1, 2, featureSpace = featureSpace)

# load model
loaded_svm = pickle.load(open("svm.pickle", "rb"))

#predict
y_predicted = loaded_svm.predict([fe.extractFeatures(text)])

if y_predicted == 1:
    print("The email is ham!")
else:
    print("The email is spam!")