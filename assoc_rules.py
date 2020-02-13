import pandas as pd
import numpy as np
import random
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data = pd.read_csv('C:/Users/jtsur/Desktop/iems_308/datasets/Dillards_POS/trnsact.csv',header = 0)
data.columns = ['sku', 'store#', 'register#','drop' , 'seq#', 'date', 'p/r', 'quantity', 'internal#', 'org_price', 'amount_charged', 'transac#', 'mic','drop1']
data = data[data['p/r'] == 'P']
data.drop_duplicates(inplace = True)
data.drop(['drop','drop1','p/r','register#','seq#','date','quantity','internal#','org_price','amount_charged','mic'],axis = 1,inplace = True)

t_counts = data['transac#'].value_counts()
t_counts = t_counts[t_counts >= 2]
data = data[data['transac#'].isin(t_counts.index)]
s_counts = data['store#'].value_counts()
s_counts = s_counts[s_counts > 100000]
data = data[data['store#'].isin(s_counts.index)]

stores = data['store#'].unique()
skus = data['sku'].unique()

random.seed = 0

t_nums = []

for i in stores:
    transacs = data[data['store#'] == i]['transac#'].unique()
    transacs = random.sample(list(transacs),400)
    t_nums.append(transacs)

flat_t_nums = [val for sublist in t_nums for val in sublist]
flat_t_nums = pd.Series(flat_t_nums)
flat_t_nums = flat_t_nums.unique()
data = data[data['transac#'].isin(flat_t_nums)] 

sku_dp = pd.read_csv('C:/Users/jtsur/Desktop/iems_308/datasets/Dillards_POS/skuinfo.csv', header = None, index_col = [0])
sku_dp.drop([2,3,4,5,6,7,8,9,10,11,12],axis = 1,inplace = True)
depts = sku_dp[1].unique()
sk_counts = data['sku'].value_counts()
sk_counts = sk_counts[sk_counts > 10]
data = data[data['sku'].isin(sk_counts.index)]
collec = []
for i in depts:
    chec = sku_dp[sku_dp[1] == i]
    chec = data[data['sku'].isin(chec.index)]
    k = chec['sku'].value_counts()
    k = k[k  > 200]
    chec = chec[chec['sku'].isin(k.index)]
    if chec.shape[0] < 6000:
        continue
    elif chec.shape[0] > 25000:
        continue
    collec.append(chec)
    
counter = 0
                 
for i in collec:                
    onehot = pd.get_dummies(i['sku'],prefix = 'sku#')
    i = pd.concat([i,onehot],axis = 1)
    i.drop(['store#','sku'],axis = 1,inplace = True)
    i = i.groupby(['transac#']).sum()
    i.replace([2,3,4,5],1,inplace = True)
    i = apriori(i, min_support=0.0001, use_colnames=True)
    i = association_rules(i, metric="lift", min_threshold=1)
    i = i[(i['lift'] >= 10) & (i['confidence'] >= .6)]
    collec[counter] = i
    counter = counter + 1

final_data = pd.concat(collec)
final_data = final_data.assign(f = final_data['lift'] * final_data['confidence'] * final_data['support'])
k = final_data.sort_values('f',ascending = False) 
k = k[0:100]
k.to_csv('C:/Users/jtsur/Desktop/iems_308/transac_data.csv')         
