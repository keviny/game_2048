import sys
import os
import random


for x in xrange(0, int(sys.argv[1])):
  f=[random.random() for x in xrange(0, 10)]
  fs = map(lambda x : "%.3f" % x, f)
  if sum(f)>5:
    print ",".join(fs)+",1"
  else:
    print ",".join(fs)+",0"
