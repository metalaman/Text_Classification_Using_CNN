import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from nltk.tokenize import word_tokenize
from collections import Counter

productTitles = list(open('../data/product_title.txt').readlines())
labels = list(open('../data/label.txt').readlines())

removingSymbols = ['=','+','>','<','`','!','@','#','$','%','^','&','*','(',')','[',']','{','}','-','_',';',':',',','.','?','/','\\','"','|','\'']

def preprocess(x):
    global removingSymbols
    a = x[0].replace('\n','').lower().strip()
    b = x[1].replace('\n','').strip()
    for ch in removingSymbols:
        if ch in a:
            a = a.replace(ch,' ')
    return (filter(lambda i: i not in ['0','1','2','3','4','5','6','7','8','9'], a), b)

data = zip(productTitles, labels)
data = map(preprocess, data)
df = pd.DataFrame(data)
df.columns = ['productTitle','label']
df.productTitle = df.productTitle.apply(lambda x: filter(lambda y: len(y) > 2, x.split())).apply(' '.join)

counter = Counter()
for i, row in df.iterrows():
    counter.update(row["productTitle"].split())

wordThreshold = 2
vocab = dict(w for w in counter.items() if w[1] >= wordThreshold)

w2idx = {}
cnt = 0
for word in vocab.keys():
    w2idx[word] = cnt
    cnt += 1
w2idx['<PAD>'] = cnt

df.productTitle = df.productTitle.apply(lambda x: filter(lambda y: y in vocab.keys(), x.split())).apply(' '.join)
df.productTitle = df.productTitle.replace('', np.nan)
df = df.dropna(subset=['productTitle'], axis=0)
maxTextLength = df.productTitle.apply(lambda x: len(x.split())).max()

#s = df.productTitle.str.len().sort_values().index
#df.reindex(s).values

def padText(x):
    global maxTextLength
    return x + (maxTextLength-len(x))*['<PAD>']

df.productTitle = df.productTitle.apply(lambda x: padText(x.split(' ')))
df.productTitle = df.productTitle.apply(lambda x: [w2idx[i] for i in x])

X = np.array([np.array(i) for i in df.productTitle.values])
y = np.array([np.array(int(i)) for i in df.label.values])

vocab["<PAD>"] = df.productTitle.str.count("<PAD>").sum()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for trainIndex, testIndex in split.split(X, y):
    XTrain, XTest = X[trainIndex], X[testIndex]
    yTrain, yTest = y[trainIndex], y[testIndex]

np.save('./Dataset/XTrain', XTrain)
np.save('./Dataset/XTest', XTest)
np.save('./Dataset/yTrain', yTrain)
np.save('./Dataset/yTest', yTest)

with open('./Utility/vocab' , 'wb') as f, open('./Utility/w2idx' , 'wb') as g:
    pickle.dump(vocab, f)
    pickle.dump(w2idx, g)
