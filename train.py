import os
from model import ProductClassifier

filterSizes = [3,4,5]
numClasses = 25
clf = ProductClassifier(filterSizes=filterSizes, numClasses=numClasses, mode='train', resume=1)
loss, inp_dict = clf.build_training_graph()
clf.train(loss, inp_dict)
