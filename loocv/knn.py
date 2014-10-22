from operator import itemgetter
import math, sys, arff, zscore, time

# this is from the solution to hw1
def get_training(training_file):
    training = []
    for row in arff.load(training_file) :
        row = list(row)
        training.append({
            'class' : row.pop(),
            'point' : row
        })
    return training

def main(args):
    training_data = get_training(args[1])
    k = 7
    test_data = training_data
    knn(training_data, test_data, k)

def knn(training_data, test_data, k, features):
    num_error = 0.0
    for te in test_data:
        ex_num = 0
        for tr in training_data:
            dist = 0.0
            for feature in features:
                dist += pow(te['point'][feature] - tr['point'][feature], 2)
            dist = pow(dist, .5)
            tr['dist'] = dist
            ex_num +=1
        sorted_training = sorted(training_data, key=itemgetter('dist'))
        
        # build neighbor count and neighbor distance dictionaries
        neighbor_count = {}    
        neighbor_dist = {}
        for example in sorted_training:
            neighbor_count[example['class']] = 0
            neighbor_dist[example['class']] = 0
        
        # calculate class neighbor count, distance for 0-k shortest distances
        for example in sorted_training[:k]: 
            neighbor_count[example['class']] += 1
            neighbor_dist[example['class']] += example['dist']
    
        # find which class(es) had the most votes
        max_neighbors = max(neighbor_count.itervalues())
        max_classes = []
        for key, value in neighbor_count.iteritems():
            if value == max_neighbors:
                max_classes.append(key)
                
        if len(max_classes) == 1:  # one class had the most votes
            classified = max_classes[0]
        else:  # need to choose class with lowest total distance
            classified = min(neighbor_dist, key=neighbor_dist.get)
                
        if classified != te['class']:
            num_error += 1
        
    #accuracy percentage
    accuracy = 1- num_error / float(len(test_data))
    return int(accuracy)


if __name__ == "__main__":
    main(sys.argv)
