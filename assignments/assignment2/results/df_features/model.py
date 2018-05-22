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
#from xgboost import XGBClassifier

import make_features_df
df = make_features_df.load_file("../../data/training_100K.csv") # Create df from the file
make_features_df.prep_dataframe(df) # make features NB: done in place!
features, target = make_features_df.create_features_and_target(df)

print(features[0])
print(target[0])

print(features[1])
print(target[1])
