from operator import itemgetter
import math, sys, arff, time, random, copy, zscore

# this is from the solution to hw1
def get_training(training_file):
    training = [] 
    i = 0
    for row in arff.load(training_file) :
        row = list(row)
        row.pop()
        training.append({
            'index' : "ex" + str(i),
            'point' : row
        })
        i += 1
    return training

def knn(k, test_point, labeled):
    num_features = len(test_point['point'])
    sorted_labeled = copy.deepcopy(labeled)
   
    for element in sorted_labeled:
        dist = get_dist(test_point, element)
        element['dist'] = dist
    
    sorted_labeled = sorted(sorted_labeled, key=itemgetter('dist'))
  
    return sorted_labeled[:k]


def calc_avg_dist(neighbors):
    dist = 0.0
    for ni in neighbors:
        dist += ni['dist']
    
    return dist / len(neighbors) 

def get_dist(x, y):
    num_features = len(x['point'])
    dist = 0.0
    for feature in range(0, num_features):
        dist += pow(x['point'][feature] - y['point'][feature], 2)
    dist = pow(dist, .5)
    return dist


def main(args):

    k = int(args[2])
    out_percent = float(args[4]) 
    data = get_training(args[5])

    # expected number of outliers
    m = int(out_percent * len(data))
  
    stdev_mean = zscore.get_stdev_mean(data)
    data = zscore.normalize_training(data, stdev_mean)
    
    for element in data:
        neighbors = knn(k, element, data)
        element['n_avg'] = round(calc_avg_dist(neighbors), 5)
    
    data = sorted(data, key=itemgetter('n_avg'))
    data.reverse()
    
    outliers = data[:m]
    
    f = open(args[6], 'w')
    text = ""
    for o in outliers:
        text += o['index'] + ":  " + str(o['n_avg']) + "\n"
    
    f.write(text)


if __name__ == "__main__":
    main(sys.argv)
