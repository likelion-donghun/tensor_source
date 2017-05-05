import pyprind
import pandas as pd
import os
pbar = pyprind.ProgBar(50000)
label = { 'pos':1, 'neg':0}
df = pd.DataFrame()
for s in ('test', 'train'):
    for i in ('pos', 'neg'):
        path = './movie/%s/%s'%(s, i)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index= True)
            pbar.update()

df.columns = ['review', 'sentiment']