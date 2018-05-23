#!/bin/bash
tail -n 100000 data/training_set_VU_DM_2014.csv > data/training_100K_T.csv &&
tail -n 250000 data/training_set_VU_DM_2014.csv > data/training_250K_T.csv &&
tail -n 500000 data/training_set_VU_DM_2014.csv > data/training_500K_T.csv &&
tail -n 1000000 data/training_set_VU_DM_2014.csv > data/training_1M_T.csv
