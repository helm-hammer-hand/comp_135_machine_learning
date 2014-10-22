#!/usr/bin/env python

# Benjamin Helm, 2/3/2014
# HW3

import sys

trainingData = [] # holds training dataset
trainingDataOne = [] # holds approach 1 training dataset
trainingDataTwo = [] # holds approach 2 training dataset

testData = [] # holds test dataset
testDataOne = []
testDataTwo = []

totalAvg = 0
avgOne = 0
avgTwo = 0

""" Reads in file from filepath, writes data to correct reference list """
def readFile(filepath, refList):
    f = open(filepath, 'r')
    for line in f:
        if line[0].isdigit():
            line = line.rstrip('\n')
            x = line.split(',')
            y = []
            y.append(int(x[0]))
            y.append(int(x[1]))
            try:
                y.append(float(x[2]))
            except ValueError:
                    y.append(x[2])
            refList.append(y)
            y.append(int(x[3]))
            y.append(x[4])
    f.close()

""" creates a training and test data set preprocessed with approach 1 """
def approachOne():
    global totalAvg
    count = 0
    for item in trainingData:
        if item[2] != '?':
            totalAvg += item[2]
            count += 1
    totalAvg = totalAvg / count
    applyAOne(trainingData, trainingDataOne)
    applyAOne(testData, testDataOne)

""" replaces '?' with correct value """
def applyAOne(dataSet, newDataSet):
    temp = []
    for item in dataSet:
        temp = list(item)
        if temp[2] == '?':
            temp[2] = totalAvg
        newDataSet.append(temp)
    writeFile(trainingDataOne, "train-pre-a1.arff")
    writeFile(testDataOne, "test-pre-a1.arff")


""" creates a training and test data set preprocessed with approach 1 """
def approachTwo():
    global avgOne, avgTwo 
   # print totalAvg
    classOne = []
    classTwo = []
    for item in trainingData:
        if item[3] == 1:
            classOne.append(item)
        elif item[3] == 2:
            classTwo.append(item)
        else:
            print "err class label"
    avgOne = calcClassAvg(classOne)
    avgTwo = calcClassAvg(classTwo)
    applyATwo()
    applyAOne(testData, testDataTwo) 
    writeFile(trainingDataTwo, "train-pre-a2.arff")
   
def calcClassAvg(classLabel):
    avg = 0
    count = 0
    for item in classLabel:
        if isinstance(item[2], float):
            avg += item[2]
            count += 1
    return avg / count


""" replaces '?' with correct value """
def applyATwo():
    temp = []
    for item in trainingData:
        temp = list(item)
        if temp[2] == '?':
            if temp[3] == 1:
              temp[2] = avgOne
            elif temp[3] == 2:
              temp[2] = avgTwo
        trainingDataTwo.append(temp)

def writeHeader(f):
    f.write("@relation HW3_TEST\n\n")
    f.write("@attribute age real\n")
    f.write("@attribute year_op real\n")
    f.write("@attribute positive_nodes real\n")
    f.write("@attribute CLASS_LABEL {1,2}\n")
    f.write("%attribute ExampleID\n\n")
    f.write("@data\n")

def writeFile(dataSet, filename):
    f = open(filename, 'w')
    writeHeader(f)
    for line in dataSet:
        f.write(",".join([str(e) for e in line]))
        f.write("\n")
    f.close()
 
def main():
    totalAvg = 0
    avgOne = 0
    avgTwo = 0
    readFile(sys.argv[1], trainingData)
    readFile(sys.argv[2], testData)
    approachOne()
    approachTwo()
            
if __name__ == "__main__":
    main()
