#!/bin/bash
cat data/training_100K.csv data/training_100K_T.csv > data/training_200K_C.csv &&
cat data/training_100K_T.csv data/training_100K.csv > data/trianing_200K_CR.csv
