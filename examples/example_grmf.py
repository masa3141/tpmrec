#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

data = pd.read_csv('data/movielens/user_ratedmovies.dat', delimiter='\t')
item_info = pd.read_csv('data/movielens/movies.dat', delimiter='\t')
itemid2name = dict(zip(item_info['id'].tolist(), item_info['title'].tolist()))

N = len(set(data['userID'].tolist()))  # number of user
M = len(set(data['movieID'].tolist()))  # number of movie

rating_matrix = np.zeros([N, M])
userid2index = {}
itemid2index = {}
userid2itemindexes = {}

for i, row in data.iterrows():
    userid = row['userID']
    itemid = row['movieID']
    rating = row['rating']
    # print userid, itemid, rating
    if userid in userid2index:
        userindex = userid2index[userid]
        userid2itemindexes[userid].append(itemid)
    else:
        userindex = len(userid2index)
        userid2index[userid] = userindex
        userid2itemindexes[userid] = [itemid]

    if itemid in itemid2index:
        itemindex = itemid2index[itemid]
    else:
        itemindex = len(itemid2index)
        itemid2index[itemid] = itemindex

    rating_matrix[userindex, itemindex] = 1.0

index2userid = {y: x for x, y in userid2index.items()}
index2itemid = {y: x for x, y in itemid2index.items()}

nonzero_row, nonzero_col = rating_matrix.nonzero()
# inds = zip(nonzero_col.tolist(), nonzero_row.tolist())
inds = [[] for i in range(N)]
for r, c in zip(nonzero_row.tolist(), nonzero_col.tolist()):
    inds[r].append(c)


import sys


sys.path.append('../tpmrec/')

from grmf import GRMF

K = 10
alpha = 0.001
lam = 0.0001
eta = 0.0000001

grmf = GRMF(rating_matrix, inds, K, alpha, lam, eta)
grmf.train(epochs=10)


for userindex in range(1000):
    userid = index2userid[userindex]
    if len(userid2itemindexes[userid]) > 20:
        continue
    pr = grmf.predict()
    user_predict = pr[userindex, :]
    top_item_indexes = np.argsort(user_predict)[::-1][:10]
    print "userid = ", userid
    for itemid in userid2itemindexes[userid]:
        print itemid, itemid2name[itemid]
    print "recommend item"
    for itemindex in top_item_indexes:
        itemid = index2itemid[itemindex]
        print itemid, itemid2name[itemid]



def read_data():
    f = open('data/movielens/user_ratedmovies.dat', 'r')
    line = f.readline().stirp('\n')
    while line:


    f.close