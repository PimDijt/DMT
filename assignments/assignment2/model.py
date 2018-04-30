import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, SelectPercentile, VarianceThreshold



with open ('feature_dict', 'rb') as fp:
    training_data = pickle.load(fp)

with open ('target_dict', 'rb') as fp:
    target_data = pickle.load(fp)

training_vec = DictVectorizer()
training_data = training_vec.fit_transform(training_data).toarray()

target_vec = DictVectorizer()
target_data = target_vec.fit_transform(target_data).toarray()
#print(len(training_data[0]))

#variance_threshold = VarianceThreshold(threshold=2)
#training_data = variance_threshold.fit_transform(training_data)
#print(len(training_data[0]))

#selector = SelectPercentile(f_classif, 10)
#selector.fit(training_data, target_data)

forest = RandomForestClassifier(n_estimators=100, random_state=1)
multi_target_forest = MultiOutputClassifier(forest)

def make_score(predictions):
    scores = []
    #first the booking

    bookings = predictions[0]
    #then the clicking
    clickings = predictions[1]

    #number
    number = len(bookings)

    for i in range(number):
        score = bookings[i][1]*5 + clickings[i][1]*1
        scores.append(score)
    
    return scores
    
scores = make_score(multi_target_forest.fit(training_data, target_data).predict_proba(training_data))
print(scores[:40])


