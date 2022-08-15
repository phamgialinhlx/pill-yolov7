import numpy as np
from numpy import genfromtxt
con = genfromtxt('/home/pill/competition/yolov7/runs/test/yolov7_45_deg_40_gen_train/confusionMatrix.csv', delimiter=',')
print(con.shape)
# for i in range(1, 108):
#   con[i, i] = 0
topn = 10
for i in range(0, 108):
  cur = con[i, :]
  ind = np.argpartition(cur, -topn)[-topn:]
  ind = ind[np.argsort(cur[ind])]
  s = "pred " + str(i) + " but not true label"
  print(s)
  for i in ind:
    if cur[i] > 0.0:
        print(i,  cur[i])

for i in range(0, 108):
  cur = con[:, i]
  ind = np.argpartition(cur, -topn)[-topn:]
  ind = ind[np.argsort(cur[ind])]
  s = "true label is " + str(i)
  print(s)
  for i in ind:
    if cur[i] > 0.0:
        print(i, cur[i])
