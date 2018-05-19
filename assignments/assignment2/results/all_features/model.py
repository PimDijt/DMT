import pickle
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

def make_daypart(hours):
    if hours >= 8 and hours < 12:
        return 0
    elif hours >= 12 and hours < 18:
        return 1
    elif hours >= 18 and hours < 24:
        return 2
    else:
        return 3 

def make_distance(km):
    if km >= 0 and km < 200:
        return 1
    elif km >= 200 and km < 1000:
        return 2
    elif km >= 1000:
        return 3
    else:
        return 0

def make_price(price):
    if price >= 0 and price < 50:
        return 1
    elif price >= 50 and price < 100:
        return 2
    elif price >= 100:
        return 3
    else:
        return 0

def make_length(length):
    if length >= 0 and length < 3:
        return 0
    elif length >= 3 and length < 8:
        return 1
    elif length >= 8 and length < 15:
        return 2
    else:
        return 3

feature_columns = [
    "srch_id",
    #"date_time",
    "site_id",
    "visitor_location_country_id",
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "prop_country_id",
    "prop_id",
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    "prop_location_score1",
    "prop_location_score2",
    #"position",
    #"price_usd",
    "promotion_flag",
    "srch_destination_id",
    #"srch_length_of_stay",
    #"srch_booking_window",
    #"srch_adults_count",
    #"srch_children_count",
    #"srch_room_count",
    "srch_saturday_night_bool",
    "srch_query_affinity_score",
    #"orig_destination_distance",
    "random_bool",
    "comp1_rate",
    "comp1_inv",
    "comp1_rate_percent_diff",
    "comp2_rate",
    "comp2_inv",
    "comp2_rate_percent_diff",
    "comp3_rate",
    "comp3_inv",
    "comp3_rate_percent_diff",
    "comp4_rate",
    "comp4_inv",
    "comp4_rate_percent_diff",
    "comp5_rate",
    "comp5_inv",
    "comp5_rate_percent_diff",
    "comp6_rate",
    "comp6_inv",
    "comp6_rate_percent_diff",
    "comp7_rate",
    "comp7_inv",
    "comp7_rate_percent_diff",
    "comp8_rate",
    "comp8_inv",
    "comp8_rate_percent_diff",
    #"gross_bookings_usd",
]
targets = [
    "booking_bool",
    "click_bool",
]

added_features = [
    "year",
    "month",
    "day",
    "day_part",
    "distance",
    "price",
    "children",
    "length",
    #"comp_score",
]

training_data = []
target_data = []
srch_id_found = False
with open('../../data/training_250K.csv', newline='') as csvfile:
    training = csv.reader(csvfile, delimiter=',')
    columns = next(training)

    for row in training:
        #if row[columns.index("booking_bool")] != "0" or row[columns.index("booking_bool")] != "0":
        extra_features = []

        #change the date, maybe add weekend?
        date_time = row[columns.index("date_time")].split()
        date = date_time[0]
        time = date_time[1]

        date = date.split("-")
        year = int(date[0])
        month = int(date[1])
        day = int(date[2])

        time = time.split(":")
        hours = int(time[0])
        minutes = int(time[1])
        seconds = int(time[2])

        day_part = make_daypart(hours)

        extra_features.extend([year, month, day, day_part])

        #change the distance
        distance = row[columns.index("orig_destination_distance")]
        if distance == 'NULL':
            distance = "-1"
        distance = make_distance(float(distance))
        extra_features.append(distance)

        #change the price
        price = row[columns.index("price_usd")]
        if price == 'NULL':
            price = "-1"
        price = make_price(float(price))
        extra_features.append(price)

        #change children
        children = int(row[columns.index("srch_children_count")])
        if children > 0:
            children = 1
        else:
            children = 0
        extra_features.append(children)

        #change stay length
        length = int(row[columns.index("srch_length_of_stay")])
        length = make_length(length)
        extra_features.append(length)

        #change competitor score
        comp_score = 0
        for i in range(1,9):
            comp_rate = row[columns.index("comp"+str(i)+"_rate")]
            if not comp_rate == "NULL":
                comp_score += int(comp_rate)
        extra_features.append(comp_score)

        #make the feature!
        result = []
        for feature in feature_columns:
            value = row[columns.index(feature)]
            if value.replace('.', '').isdigit():
                result.append(float(value))
            else:
                result.append(-1)

        for feature in added_features:
            result.append(extra_features[added_features.index(feature)])

        training_data.append(result)

        #make target!
        target_result = []
        for target in targets:
            target_result.append(int(row[columns.index(target)]))
        target_data.append(target_result)
	
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
        #print("Fold: {}".format(i+1))
        training_data = []
        training_targets = []
        test_data = []
        test_targets = []

        cur_srch_id = -1
        count = 0
        iterator = 0
        while count < i*slice_size:
            if data[iterator][0] != cur_srch_id:
                cur_srch_id = data[iterator][0]
                count += 1
                if count == slice_size:
                    break
            training_data.append(data[iterator])
            training_targets.append(targets[iterator])
            iterator += 1

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
        if test_data[i][0] != cur_srch_id:
            #begin new search query
            cur_srch_id = test_data[i][0]
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
    score = 0
    for i in range(0, len(targets)):
        if targets[i][0] == 1:
            score += (2**5 - 1) / math.log2(i+1+1)
        else:
            score += (2**(targets[i][1] * 1) - 1 ) / math.log2(i+1+1)
    return score

def calc_max(targets):
    click_count = 0
    book_count = 0
    for item in targets:
        if item[1] == 1:
            click_count += 1
        if item[0] == 1:
            book_count = 1

    if book_count > 0:
        score = (2**5 -1)
    elif click_count > 0:
        score = (2**1 -1)
        click_count -= 1
    else:
        score = 0
    for i in range(0, click_count):
        score += (2**1 -1) / math.log2(i+1+1)
    return score

parameters = [20,40,60,80,100]

print("Random Forest:")
for p in parameters:
    print("{} estimators".format(p))
    forest = MultiOutputClassifier(RandomForestClassifier(n_estimators=p, random_state=1))
    cross_validate(forest, training_data, target_data, search_amount, n_folds=10)

print("Ada Boost:")
for p in parameters:
    print("Ada boost, estimators: {}".format(p))
    ada = MultiOutputClassifier(AdaBoostClassifier(n_estimators=p, random_state=1))
    cross_validate(ada, training_data, target_data, search_amount, n_folds=10)

print("Extra Tree:")
for p in parameters:
    print("Ada boost, estimators: {}".format(p))
    trees = MultiOutputClassifier(ExtraTreesClassifier(n_estimators=p, random_state=1))
    cross_validate(trees, training_data, target_data, search_amount, n_folds=10)

print("Gradient Boosting:")
for p in parameters:
    print("Gradient boost, estimators: {}".format(p))
    grboost = MultiOutputClassifier(GradientBoostingClassifier(n_estimators=p, random_state=1))
    cross_validate(grboost, training_data, target_data, search_amount, n_folds=10)

'''
for l in layers:
    t = ()
    for i in range(0,l):
        t += (20,)
    print("Neural net, {} layers of size 20:".format(l))
    net = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=t, random_state=1))
    cross_validate(net, training_data, target_data, search_amount, n_folds=10)
'''
