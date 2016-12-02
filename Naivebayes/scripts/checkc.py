import sys
import json


with open(sys.argv[1]) as rd:
  data = json.load(rd)

countp, countn = 0, 0

for i in range(len(data)):
  print i
  if data[i]['stars'] > 3.5:
    countp += 1
  else:
    countn += 1


print 'Pos', countp
print 'neg', countn
