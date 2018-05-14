import time
import os
import csv

def write_submission(recommendations, submission_file=None):
    rows = [(srch_id, prop_id) for srch_id, prop_id, rank_float in sorted(recommendations, key=itemgetter(0,2))]
    fname = "output_%d.csv".format( int(time.time()) ) if submission_file == None else submission_file

    writer = csv.writer(open(fname, "w"), lineterminator="\n")
    writer.writerow(("SearchId", "PropertyId"))
    writer.writerows(rows)