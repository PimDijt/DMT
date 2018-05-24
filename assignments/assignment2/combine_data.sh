#!/bin/bash
cat data/training_100K.csv data/training_100K_T.csv > data/training_200K_C.csv &&
head -n 1 data/training_100K.csv >> data/training_200K_CR.csv &&
cat data/training_100K_T.csv >> data/training_200K_CR.csv &&
tail -n +2 data/training_100K.csv >> data/training_200K_CR.csv
