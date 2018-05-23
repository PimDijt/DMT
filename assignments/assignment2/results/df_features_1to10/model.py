import numpy as np
import sys
import math
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, SelectPercentile, VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import clone
from xgboost import XGBClassifier

import make_features_df
df = make_features_df.load_file("../../data/training_250K.csv") # Create df from the file
make_features_df.prep_dataframe(df) # make features NB: done in place!
training_data, target_data = make_features_df.create_features_and_target(df)



#Find how many unique searchers are in the data.
search_amount = 0
cur_srch_id = -1
for item in training_data:
    if item[0] != cur_srch_id:
        cur_srch_id = item[0]
        search_amount += 1

def cross_validate(model, data, targets, search_amount, n_folds=10):
    if n_folds == 1:
        slice_size = search_amount/10
    else:
        slice_size = search_amount / n_folds
    scores = []
    for i in range(0,n_folds):
        training_data = []
        training_targets = []
        test_data = []
        test_targets = []

        cur_srch_id = -1
        count = 0
        iterator = 0

        #take "left" of the test part
        while count < i*slice_size:
            if data[iterator][0] != cur_srch_id:
                cur_srch_id = data[iterator][0]
                count += 1
                if count == slice_size:
                    break
            training_data.append(data[iterator][1:])
            training_targets.append(targets[iterator])
            iterator += 1

        #take the test part
        count = 0
        while count < slice_size and iterator < len(data):
            if data[iterator][0] != cur_srch_id:
                cur_srch_id = data[iterator][0]
                count += 1
                if count == slice_size:
                    break
            test_data.append(data[iterator])
            test_targets.append(targets[iterator])
            iterator += 1

        #take "right" of the test part
        while iterator < len(data):
            training_data.append(data[iterator][1:])
            training_targets.append(targets[iterator])
            iterator += 1


        fold_model = clone(model)
        fold_model.fit(training_data, training_targets)

        score = assess_model(fold_model, test_data, test_targets)
        scores.append(score)
    for s in scores:
        print("{0:.2f}".format(s), end=" ")
    print("\nAverage: {0:.2f}\n".format(sum(scores)/len(scores)))

def assess_model(model, test_data, test_targets):
    cur_srch_id = -1
    cur_search = []
    cur_targets = []
    scores = []
    for i in range(0, len(test_data)):
        if test_data[i][0] != cur_srch_id:
            #begin new search query
            cur_srch_id = test_data[i][0]
            if i != 0:
                score = assess_search(model, cur_search, cur_targets)
                scores.append(score)
            cur_search = [test_data[i][1:]]
            cur_targets = [test_targets[i]]
        else:
            cur_search.append(test_data[i][1:])
            cur_targets.append(test_targets[i])
    return sum(scores)/len(scores)

def assess_search(model, search, targets):
    results = model.predict_proba(search)
    scores = []
    book_proba = results[0]
    click_proba = results[1]
    for i in range(0, len(book_proba)):
        score = book_proba[i][1]*10 + click_proba[i][1]*1
        scores.append(score)

    #old_scores = scores
    #old_targets = targets

    scores, targets = (list(t) for t in zip(*sorted(zip(scores, targets), reverse=True)))

    new_targets = sorted(targets, key = lambda x: (x[0], x[1]), reverse=True)


    my_score = calc_score(targets)

    #max_score = calc_max(targets)

    max_score2 = calc_score(new_targets)
    ndcg = my_score / max_score2 if max_score2 > 0 else 0
    # if max_score != max_score2:
    #     print(my_score, max_score, ndcg, max_score2)
    #     for i in range(0, len(scores)):
    #         print(old_scores[i], scores[i], old_targets[i], targets[i], new_targets[i])
    #     exit()
    return ndcg

def calc_score(targets):
    score = 0
    for i in range(0, len(targets)):
        if targets[i][0] == 1:
            score += (2**5 - 1) / math.log2(i+1+1)
        elif targets[i][1] == 1:
            score += (2**1 - 1 ) / math.log2(i+1+1)
    return score

def calc_max(targets):
    click_count = 0
    book_count = 0
    score = 0
    for item in targets:
        if item[1] == 1:
            click_count += 1
        if item[0] == 1:
            book_count = 1
    if book_count > 0:
        score = (2**5) - 1
        click_count -= 1
    elif click_count > 0:
        score = (2**1) - 1
        click_count -= 1
    else:
        score = 0
    for i in range(0, click_count):
        score += (2**1 -1) / math.log2(i+1+1)
    return score

parameters = [20,40,60,80,100]

print("XGBoost:")
for p in parameters:
    print("{} estimators".format(p))
    xgboost = MultiOutputClassifier(XGBClassifier(n_estimators=p, learning_rate=0.01, n_jobs=-1))
    cross_validate(xgboost, training_data, target_data, search_amount, n_folds=10)

print("Random Forest:")
for p in parameters:
    print("{} estimators".format(p))
    forest = MultiOutputClassifier(RandomForestClassifier(n_estimators=p, n_jobs=-1))
    cross_validate(forest, training_data, target_data, search_amount, n_folds=10)

print("Ada Boost:")
for p in parameters:
    print("{} estimators".format(p))
    ada = MultiOutputClassifier(AdaBoostClassifier(n_estimators=p, learning_rate=0.01))
    cross_validate(ada, training_data, target_data, search_amount, n_folds=10)

print("Gradient Boosting:")
for p in parameters:
    print("{} estimators".format(p))
    grboost = MultiOutputClassifier(GradientBoostingClassifier(n_estimators=p, learning_rate=0.01))
    cross_validate(grboost, training_data, target_data, search_amount, n_folds=10)
