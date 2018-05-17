#!/bin/bash
head -n 100000 data/training_set_VU_DM_2014.csv > data/training_100K.csv &&
head -n 250000 data/training_set_VU_DM_2014.csv > data/training_250K.csv &&
head -n 500000 data/training_set_VU_DM_2014.csv > data/training_500K.csv &&
head -n 1000000 data/training_set_VU_DM_2014.csv > data/training_1M.csv
