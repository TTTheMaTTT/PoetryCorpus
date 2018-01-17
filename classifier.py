# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:55:57 2017

@author: Sergey
"""

import pymorphy2
import collections
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

full_dict = []
morph = pymorphy2.MorphAnalyzer()
n = 0
dict_of_friendship = collections.Counter()
dict_of_home = collections.Counter()
dict_of_love = collections.Counter()
for name in ["О родине.txt", "Олюбви1.txt", "О дружбе.txt"]:
    f = open(name, "r", encoding = "utf-8")
    if name == "О родине.txt":
        d = dict_of_home
    if name == "Олюбви1.txt":
        d = dict_of_love
    if name == "О дружбе.txt":
        d = dict_of_friendship
    k = 0
    f = f.read()
    f = f.splitlines()
    for line in f:
        if "<text" in line:
            continue
        if "text>" in line:
            k += 1
            n += 1
            continue
        for word in line.split():
            lema = morph.parse(word)[0].normal_form
            d[lema] += 1        
#    print(name, k, d.most_common(100))
    words = list(dict(d.most_common(80)))
    for word in words:
        if word not in full_dict:
            full_dict.append(word)
    print(name, k)
matrix = np.zeros((n, len(full_dict)))
labels = np.zeros((n, 3))

n = 0
for name in ["О родине.txt", "Олюбви1.txt", "О дружбе.txt"]:
    if name == "О родине.txt":
        i = 0
    if name == "Олюбви1.txt":
        i = 1
    if name == "О дружбе.txt":
        i = 2
    f = open(name, "r", encoding = "utf-8")
    f = f.read()
    f = f.splitlines()
    k = 0
    for line in f:
        if "<text" in line:
            continue
        if "text>" in line:
            labels[n][i] = 1
            for j in range(len(full_dict)):
                matrix[n][j] = matrix[n][j] / k
            k = 0
            n += 1
            continue
        for word in line.split():
            k += 1
            lema = morph.parse(word)[0].normal_form
            if lema in full_dict:
                matrix[n][full_dict.index(lema)] += 1

c = np.zeros((len(full_dict), 3, 3))
for pr in range(len(full_dict)):
    sum_i = [0, 0, 0]
    k = [0, 0, 0]
    total_sum = 0
    for i in range(n):
        total_sum += matrix[i][pr]
        for j in range(3):
            if labels[i][j]:
                sum_i[j] += matrix[i][pr]
                k[j] += 1
    for i in range(3):
        sum_i[i] = sum_i[i] / k[i]
    for i in range(3):
        if sum_i[i] > 0:
            for j in range(3):
                c[pr][i][j] = sum_i[j] / sum_i[i]
#    print(full_dict[pr], c)
                
feature_of_home = []
feature_of_friendship = []
feature_of_love = []
#feature_of_friendship = dict_of_friendship.most_common(200)
#feature_of_home = dict_of_home.most_common(200)
#feature_of_love = dict_of_love.most_common(200)
for i in range(len(full_dict)):
    if (c[i][0][0] + c[i][0][1] + c[i][0][2] < 2.6) and (c[i][0][0] + c[i][0][1] + c[i][0][2] >= 1) and c[i][0][1] < 0.9 and c[i][0][2] < 0.9:
        feature_of_home.append(i)

for i in range(len(full_dict)):
    if (c[i][1][0] + c[i][1][1] + c[i][1][2] < 2.6) and (c[i][1][0] + c[i][1][1] + c[i][1][2] >= 1) and c[i][1][0] < 0.9 and c[i][1][2] < 0.9:
        feature_of_love.append(i)
        
for i in range(len(full_dict)):
    if (c[i][2][0] + c[i][2][1] + c[i][2][2] < 2.4) and (c[i][2][0] + c[i][2][1] + c[i][2][2] >= 1) and c[i][2][1] < 0.9 and c[i][2][0] < 0.9:
        feature_of_friendship.append(i)
        

print("home", "/n")
for i in range(len(feature_of_home)):
    print(full_dict[feature_of_home[i]])
print("love", "/n")
for i in range(len(feature_of_love)):
    print(full_dict[feature_of_love[i]])
print("family", "/n")
for i in range(len(feature_of_friendship)):
    print(full_dict[feature_of_friendship[i]])

1

For_Home = np.zeros((n, len(feature_of_home)))
For_Love = np.zeros((n, len(feature_of_love)))
For_Family = np.zeros((n, len(feature_of_friendship)))

for i in range(len(full_dict)):
    if i in feature_of_home:
        for j in range(n):
            For_Home[j][feature_of_home.index(i)] = matrix[j][i]
    if i in feature_of_love:
        for j in range(n):
            For_Love[j][feature_of_love.index(i)] = matrix[j][i]
    if i in feature_of_friendship:
        for j in range(n):
            For_Family[j][feature_of_friendship.index(i)] = matrix[j][i]


X_train, X_test, y_train, y_test = train_test_split(For_Home, labels[:,0], test_size=0.2, random_state=22)
rf_home = RandomForestRegressor(n_estimators=10).fit(X_train, y_train)
p = rf_home.predict(X_test)
fp = 0
fn = 0
all = 0
for i in range(len(y_test)):
    if int(p[i] >= 0.5) != y_test[i]:
        if int(p[i] >= 0.5) == 0:
            fn += 1
        if int(p[i] >= 0.5) == 1:
            fp += 1
    if y_test[i] == 1:
        all += 1
print(fp, fn, all)

X_train, X_test, y_train, y_test = train_test_split(For_Love, labels[:,1], test_size=0.2, random_state=22)
rf_love = RandomForestRegressor(n_estimators=10).fit(X_train, y_train)
p = rf_love.predict(X_test)
fp = 0
fn = 0
all = 0
for i in range(len(y_test)):
    if int(p[i] >= 0.5) != y_test[i]:
        if int(p[i] >= 0.5) == 0:
            fn += 1
        if int(p[i] >= 0.5) == 1:
            fp += 1
    if y_test[i] == 1:
        all += 1
print(fp, fn, all)

X_train, X_test, y_train, y_test = train_test_split(For_Family, labels[:,2], test_size=0.2, random_state=22)
rf_friend = RandomForestRegressor(n_estimators=10).fit(X_train, y_train)
p = rf_friend.predict(X_test)
print(p)
fp = 0
fn = 0
all = 0
for i in range(len(y_test)):
    if int(p[i] >= 0.5) != y_test[i]:
        if int(p[i] >= 0.5) == 0:
            fn += 1
        if int(p[i] >= 0.5) == 1:
            fp += 1
    if y_test[i] == 1:
        all += 1
print(fp, fn, all)

k_of_love = 0
k_of_home = 0
k_of_friendship = 0

n = 0
exit_1 = 0
morph = pymorphy2.MorphAnalyzer()
tree = os.walk(r"C:\Users\Sergey\Desktop\abbyy\corpora\a")
dict_of_friendship1 = collections.Counter()
dict_of_home1 = collections.Counter()
dict_of_love1 = collections.Counter()
for d, dirs, files in tree:
    for f in files:
        n += 1
        if exit_1 == 2:
            continue
        For_Home = np.zeros((1, len(feature_of_home)))
        For_Love = np.zeros((1, len(feature_of_love)))
        For_Family = np.zeros((1, len(feature_of_friendship)))
        path = os.path.join(d,f)
        text = open(path, "r", encoding = "utf-8").read()
        text = text.splitlines()
        k = 0
        for line in text:
            if "id" in line:
                continue
            if "url" in line:
                continue
            if "author" in line:
                continue
            if "/a" in line:
                continue
            if "/div" in line:
                continue
            for word in line.split():
                k +=1
                word = morph.parse(word)[0].normal_form
                if word in full_dict:
                    if full_dict.index(word) in feature_of_home:
                        For_Home[0][feature_of_home.index(full_dict.index(word))] += 1
                    if full_dict.index(word) in feature_of_love:
                        For_Love[0][feature_of_love.index(full_dict.index(word))] += 1
                    if full_dict.index(word) in feature_of_friendship:
                        For_Family[0][feature_of_friendship.index(full_dict.index(word))] += 1
        For_Home = For_Home / k
        For_Love = For_Love / k
        For_Family = For_Family / k
        k_of_home += int(rf_home.predict(For_Home) >=0.5)
        k_of_love += int(rf_love.predict(For_Love) >= 0.5)
        k_of_friendship += int(rf_friend.predict(For_Family) >= 0.5)
        if rf_home.predict(For_Home) >=0.5:
            print(text)
            exit_1 += 1
        
print(k_of_home, k_of_love, k_of_friendship, n)