import csv
import sys
import pickle

def make_daypart(hours):
    if hours >= 8 and hours < 12:
        return "morning"
    elif hours >= 12 and hours < 18:
        return "day"
    elif hours >= 18 and hours < 24:
        return "evening"
    else:
        return "night"

def make_distance(km):
    if km >= 0 and km < 200:
        return "close"
    elif km >= 200 and km < 1000:
        return "medium"
    elif km >= 1000:
        return "far"
    else:
        return "unknown"

def make_price(price):
    if price >= 0 and price < 50:
        return "cheap"
    elif price >= 50 and price < 100:
        return "medium"
    elif price >= 100:
        return "expensive"
    else:
        return "unknown"

def make_length(length):
    if length >= 0 and length < 3:
        return "days"
    elif length >= 3 and length < 8:
        return "week"
    elif length >= 8 and length < 15:
        return "weeks"
    else:
        return "month"

feature_columns = [
    "srch_id",
    #"date_time",
    "site_id",
    "visitor_location_country_id",
    "visitor_hist_starrating",
    #"visitor_hist_adr_usd",
    "prop_country_id",
    "prop_id",
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    #"prop_location_score1",
    #"prop_location_score2",
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
    #"srch_query_affinity_score",
    #"orig_destination_distance",
    "random_bool",
    #"comp1_rate",
    #"comp1_inv",
    #"comp1_rate_percent_diff",
    #"comp2_rate",
    #"comp2_inv",
    #"comp2_rate_percent_diff",
    #"comp3_rate",
    #"comp3_inv",
    #"comp3_rate_percent_diff",
    #"comp4_rate",
    #"comp4_inv",
    #"comp4_rate_percent_diff",
    #"comp5_rate",
    #"comp5_inv",
    #"comp5_rate_percent_diff",
    #"comp6_rate",
    #"comp6_inv",
    #"comp6_rate_percent_diff",
    #"comp7_rate",
    #"comp7_inv",
    #"comp7_rate_percent_diff",
    #"comp8_rate",
    #"comp8_inv",
    #"comp8_rate_percent_diff",
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
    "comp_score",
]

training_data = []
target_data = []
with open('data/training_100K.csv', newline='') as csvfile:
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
            children = "yes"
        else:
            children = "no"
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
        result_dict = {}
        for feature in feature_columns:
            value = row[columns.index(feature)]
            if value.replace('.', '').isdigit():
                result_dict[feature] = float(value)
            else:
                result_dict[feature] = value

        for feature in added_features:
            result_dict[feature] = extra_features[added_features.index(feature)]

        training_data.append(result_dict)

        #make target!
        target_dict = {}
        for target in targets:
            target_dict[target] = int(row[columns.index(target)])

        target_data.append(target_dict)


with open('feature_dict_100K.dict', 'wb') as fp:
    pickle.dump(training_data, fp)

with open('target_dict_100K.dict', 'wb') as fp:
    pickle.dump(target_data, fp)
