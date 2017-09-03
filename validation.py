import os
from model import ProductClassifier

filterSizes = [3,4,5]
numClasses = 25
clf = ProductClassifier(filterSizes=filterSizes, numClasses=numClasses, mode='validation')
accuracy, inputDict = clf.build_validation_graph()
clf.validation(accuracy, inputDict)
