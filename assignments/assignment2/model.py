import pickle
import numpy as np
import sys
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, SelectPercentile, VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import clone

with open ('feature_dict_100K.dict', 'rb') as fp:
    training_data = pickle.load(fp)

with open ('target_dict_100K.dict', 'rb') as fp:
    target_data = pickle.load(fp)

training_vec = DictVectorizer()
training_data = training_vec.fit_transform(training_data).toarray()

target_vec = DictVectorizer()
target_data = target_vec.fit_transform(target_data).toarray()

search_amount = 0
cur_srch_id = -1
for item in training_data:
    if item[training_vec.feature_names_.index('srch_id')] != cur_srch_id:
        cur_srch_id = item[training_vec.feature_names_.index('srch_id')]
        search_amount += 1

def cross_validate(model, data, targets, search_amount, n_folds=10):
    if n_folds == 1:
        slice_size = search_amount/10
    else:
        slice_size = search_amount / n_folds
    scores = []
    for i in range(0,n_folds):
        #print("Fold: {}".format(i+1))
        training_data = []
        training_targets = []
        test_data = []
        test_targets = []

        cur_srch_id = -1
        count = 0
        iterator = 0
        while count < i*slice_size:
            if data[iterator][training_vec.feature_names_.index('srch_id')] != cur_srch_id:
                cur_srch_id = data[iterator][training_vec.feature_names_.index('srch_id')]
                count += 1
                if count == slice_size:
                    break
            training_data.append(data[iterator])
            training_targets.append(targets[iterator])
            iterator += 1

        count = 0
        while count < slice_size and iterator < len(data):
            if data[iterator][training_vec.feature_names_.index('srch_id')] != cur_srch_id:
                cur_srch_id = data[iterator][training_vec.feature_names_.index('srch_id')]
                count += 1
                if count == slice_size:
                    break
            test_data.append(data[iterator])
            test_targets.append(targets[iterator])
            iterator += 1

        while iterator < len(data):
            training_data.append(data[iterator])
            training_targets.append(targets[iterator])
            iterator += 1

        fold_model = clone(model)
        fold_model.fit(training_data, training_targets)

        score = assess_model(fold_model, test_data, test_targets)
        scores.append(score)
    for s in scores:
        print("{0:.2f}".format(s), end=" ")
    print("\nAverage: {0:.2f}\n".format(sum(scores)/len(scores)))



def assess_model(multi_target_forest, test_data, test_targets):
    cur_srch_id = -1
    cur_search = []
    cur_targets = []
    scores = []
    for i in range(0, len(test_data)):
        if test_data[i][training_vec.feature_names_.index('srch_id')] != cur_srch_id:
            #begin new search query
            cur_srch_id = test_data[i][training_vec.feature_names_.index('srch_id')]
            if i != 0:
                score = assess_search(multi_target_forest, cur_search, cur_targets)
                scores.append(score)
            cur_search = [test_data[i]]
            cur_targets = [list(test_targets[i])]
        else:
            cur_search.append(test_data[i])
            cur_targets.append(list(test_targets[i]))
    #print("Score: {}".format(sum(scores)/len(scores)))
    return sum(scores)/len(scores)

def assess_search(multi_target_forest, search, targets):
    results = multi_target_forest.predict_proba(search)
    scores = []
    book_proba = results[0]
    click_proba = results[1]
    for i in range(0, len(book_proba)):
        score = book_proba[i][1]*5 + click_proba[i][1]*1
        scores.append(score)

    scores, targets = (list(t) for t in zip(*sorted(zip(scores, targets), reverse=True)))


    my_score = calc_score(targets)
    max_score = calc_max(targets)

    ndcg = my_score / max_score
    return ndcg

def calc_score(targets):
    if targets[0][0] == 1:
        score = 5
    else:
        score = targets[0][1] * 1
    for i in range(1, len(targets)):
        if targets[i][1] == 1:
            score += 5 / math.log2(i+1)
        else:
            score += targets[i][1] * 1 / math.log2(i+1)
    return score

def calc_max(targets):
    click_count = -1
    for item in targets:
        if item[1] == 1:
            click_count += 1

    score = 5
    for i in range(1, click_count+1):
        score += 1 / math.log2(i+1)
    return score

parameters = [20, 40, 60, 80, 100]

for p in parameters:
    print("Random Tree:")
    print("{} estimators:".format(p))
    forest = MultiOutputClassifier(ExtraForestClassifier(n_estimators=p, random_state=1))
    cross_validate(forest, training_data, target_data, search_amount, n_folds=10)
