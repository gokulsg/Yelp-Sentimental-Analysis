# import modules
import json
import glob
import os
import sys
import time
 
# start process time
start = time.clock()
 
# read in yelp data
# yelp_files = "%s/data/yelp_academic_dataset_review.json" % os.getcwd()
yelp_data = []
with open(sys.argv[1]) as f:
  for line in f:
    yelp_data.append(json.loads(line))
 
# extract user rating information
counter = 0
user_rating = []
for item in yelp_data:
  counter += 1
  print counter
  temp = {}
  temp['stars'] = item[u'stars']
  temp['text'] = item[u'text']
  user_rating.append(temp)
  if counter == 100000:
    break

with open('outputfile.json', 'w') as fout:
    json.dump(user_rating, fout)

elapsed = (time.clock() - start)