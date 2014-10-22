import sys, math

def test_zscore(ary):
    test_ary = []
    for line in ary:
        newdict = {}
        newdict['point'] = line
        test_ary.append(newdict)
    stdev_mean = get_stdev_mean(test_ary)
    print(normalize_training(test_ary, stdev_mean))

def get_stdev_mean(dataset):
    """returns 2d list with height of the number of features, column 1 feature
       mean and column 2 feature standard deviation """
    num_features = len(dataset[0]['point'])
    stdev_mean = []

    for feature in range (0, num_features):
        new = []
        column = []
        for datapoint in dataset:
            column.append(datapoint['point'][feature])
    #    print column
        mean = calc_mean(column)
        stdev = calc_stdev(column, mean)
        new.append(mean)
        new.append(stdev)
        stdev_mean.append(new)
   # print "\n"
   # print stdev_mean
    return stdev_mean

def calc_mean(data):
    mean = 0.0
    for element in data:
        mean += element
    return mean / len(data)

def calc_stdev(data, mean):
    stdev = 0
    for element in data:
        stdev += pow(element - mean, 2)
    stdev = stdev / (len(data) - 1)
    return pow(stdev, .5)

def un_zscore(z_value, stdev, mean):
    return z_value * stdev + mean 

def normalize_training(dataset, stdev_mean):
    width = len(dataset[0]['point'])
    for datapoint in dataset:
        for feature in range (0, width):
            xij = datapoint['point'][feature]  
            uj = stdev_mean[feature][0]
            sj = stdev_mean[feature][1]
            datapoint['point'][feature] = (xij - uj) / sj
    #print dataset
    #print "\n"
    return dataset

def normalize_test(point, stdev_mean):
    for feature in range (0, len(point)):
        xij = point[feature]  
       # print xij
        uj = stdev_mean[feature][0]
        sj = stdev_mean[feature][1]
        point[feature] = (xij - uj) / sj
    return point
