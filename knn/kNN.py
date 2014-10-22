#!/usr/bin/env python

"""
Benjamin Helm, January 22, 2014
Comp 135
Assignment 1
"""

from operator import itemgetter
import sys

trainingData = [] # holds training dataset
testData = [] # holds test dataset
k = 9 # maximum number of nearest neighbors to consider


""" k is the number of nearest neighbors to consider, itemNum is the index 
    of the current datapoint in testData.  """
def appendCategory(k, itemNum):
    maxCategories = []
    neighborCount = {'versicolor':0, 'virginica':0, 'setosa':0}
    neighborDist = {'versicolor':0, 'virginica':0, 'setosa':0}
    for i in range(0, k):
            neighborCount[trainingData[i][4]] += 1
            neighborDist[trainingData[i][4]] += trainingData[i][6]
    tempMax = max(neighborCount.iteritems(), key=itemgetter(1))[0]
    for category, count in neighborCount.iteritems():
        if count == neighborCount[tempMax]:
            maxCategories.append(category)
    if len(maxCategories) == 1:
        testData[itemNum].append(maxCategories[0])
    else:
        minDist = float("inf")
        currentCat = ""
        for cat in maxCategories:
            if neighborDist[cat] < minDist:
                minDist = neighborDist[cat]
                currentCat = cat
        testData[itemNum].append(currentCat)

""" filepath is the name of the text file to read from, reflist is the dataset
   where the data should be stored """
def readFile(filepath, refList):
    f = open(filepath, 'r')
    for line in f:
        if line[0].isdigit():
            line = line.rstrip('\n')
            x = line.split(',')
            y = []
            for element in x:
                try:
                    y.append(float(element))
                except ValueError:
                    y.append(element)
            refList.append(y)
    f.close()

def writeHeader(f):
    f.write("@relation HW1_TEST\n\n")
    f.write("@attribute sepal_length real\n")
    f.write("@attribute sepal_width real\n")
    f.write("@attribute petal_length real\n")
    f.write("@attribute petal_width real\n")
    f.write("%attribute ExampleID\n\n")
    f.write("@data\n")

def writeFile():
    f = open(sys.argv[3], 'w')
    writeHeader(f)
    for line in testData:
        f.write(",".join([str(e) for e in line]))
        f.write("\n")
    f.close()

""" k is the maximum number of nearest neighbors to consider """
def calcKNN(k):
    itemNum = 0;
    for te in testData:
        for tr in trainingData:
            dist = pow((te[0] - tr[0]),2) + pow((te[1] - tr[1]),2) + \
                   pow((te[2] - tr[2]),2) + pow((te[3] - tr[3]),2)
            dist = pow(dist, .5)
            tr[6] = dist
        trainingData.sort(key=itemgetter(6))
        for kval in range(1,k+1,2):
            appendCategory(kval, itemNum)
        itemNum += 1

def main():
    readFile(sys.argv[1], trainingData)
    readFile(sys.argv[2], testData)
    for element in trainingData:
        element.append(0)
    calcKNN(k)
    writeFile()
        
if __name__ == "__main__":
    main()
