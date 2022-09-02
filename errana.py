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

import os
import numpy as np
from loadOCR import load_OCR_res, take_id_from_pres

OCR_res = load_OCR_res()
directory = '/home/pill/competition/yolov7/vaipeFixGen/labels/'
files = os.listdir(directory)
con = np.zeros((109, 109))
for file in files:
  path = directory + file
  f = open(path, 'r')
  lines = f.readlines()
  name_pres = take_id_from_pres(path)
  if name_pres in OCR_res.keys():
    id_potential_in_pres = np.array(OCR_res[name_pres]).astype(int)
    for line in lines:
      cls = int(line.split()[0])
      for id in id_potential_in_pres:
        con[id][cls] = con[id][cls] + 1
  
print(con)
