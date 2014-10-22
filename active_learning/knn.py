from operator import itemgetter
#import numpy
#import matplotlib.pyplot as plt
import math, sys, arff, time, random, copy

# this is from the solution to hw1
def get_training(training_file):
    training = [] 
    i = 0
    for row in arff.load(training_file) :
        row = list(row)
        training.append({
            'class' : row.pop(),
            'point' : row,
            'index' : i
        })
        i += 1
    return training

def get_validation(dataset, v):
    validation = []
    for i in range (0, v):
        #random.seed(100)
        index = random.randint(0, len(dataset) - 1)
        validation.append(dataset[index])
        dataset.pop(index)
    
    return validation

def get_labeled(dataset, k):
    labeled = []
    classes = []

    for member in dataset:
        if member['class'] not in classes:
            classes.append(member['class'])

    num_classes = float(len(classes))
    x = int(math.ceil(float(k)/num_classes))
    
    for c in classes:
        class_members = []
        for ex in dataset:
            if ex['class'] == c:
                class_members.append(ex)
        for i in range (0, x):
            #random.seed(500)
            temp = random.randint(0, len(class_members)-1)
            labeled.append(class_members[temp])
            dataset.pop(temp)
            
    return labeled
    

def calc_vote(test_point, ni, sigma):
    dist = get_dist(test_point, ni)
    dist = pow(dist, 2)
    dist *= -1
    dist = dist / (2 * sigma * sigma)
    return pow(math.e, dist)
    
def validate(labeled, validation, k, sigma):
    for test_point in validation:
        # get k nearest neighbors
        total_votes = {}
        total_dists = {}
        neighbors = knn(k, test_point, labeled)
       
        # count up total votes and distances
        for ni in neighbors:
            #print ni['index']
            ni['vote'] = calc_vote(test_point, ni, sigma)
            if ni['class'] in total_votes:
                total_votes[ni['class']] += ni['vote']
            else:
                total_votes[ni['class']] = ni['vote']
            if ni['class'] in total_dists:
                total_dists[ni['class']] += ni['dist']
            else:
                total_dists[ni['class']] = ni['dist']
        
        # lala
        max_vote = max(total_votes.itervalues())
        max_vote_classes = []
        for key, value in total_votes.iteritems():
            if value == max_vote:
                max_vote_classes.append(key)

        if len(max_vote_classes) == 1: # one clss had the most votes
            guess = max_vote_classes[0]
        else:
            min_dist_class = -1
            min_dist = float("inf")
            for mvc in max_vote_classes:
                if total_dists[mvc] < min_dist:
                    min_dist = total_dists[mvc]
                    min_dist_class = mvc
            guess = min_dist_class
                    
        test_point['guess'] = guess
        

    # calculate accuracy
    num_correct = 0.0
    for test_point in validation:
        if test_point['class'] == test_point['guess']:
            num_correct += 1
    
    return num_correct / len(validation)
    

def get_dist(x, y):
    num_features = len(x['point'])
    dist = 0.0
    for feature in range(0, num_features):
        dist += pow(x['point'][feature] - y['point'][feature], 2)
    dist = pow(dist, .5)
    return dist

def knn(k, test_point, labeled):
    num_features = len(test_point['point'])
    sorted_labeled = copy.deepcopy(labeled)
   
    for element in sorted_labeled:
        dist = get_dist(test_point, element)
        element['dist'] = dist
    
    sorted_labeled = sorted(sorted_labeled, key=itemgetter('dist'))
  
    return sorted_labeled[:k]


def calc_uncertainties(u_labeled, u_unlabeled, sigma, k):
    uncertain_unlabeled = copy.deepcopy(u_unlabeled)
    for test_point in uncertain_unlabeled:
        total_votes = {}
        neighbors = knn(k, test_point, u_labeled)
        
        for ni in neighbors:
            ni['vote'] = calc_vote(test_point, ni, sigma)
            if ni['class'] in total_votes:
                total_votes[ni['class']] += ni['vote']
            else: 
                total_votes[ni['class']] = ni['vote']
        
        sorted_votes = sorted(total_votes.values())
        sorted_votes.reverse()
        sorted_votes.append(0)

        certainty = sorted_votes[0] - sorted_votes[1]
       
        test_point['certainty'] = certainty

    uncertain_unlabeled = sorted(uncertain_unlabeled, 
                                 key=itemgetter('certainty'))

    for element in uncertain_unlabeled:
        del element['certainty']
    
    return uncertain_unlabeled

def calc_stdev(data, mean):
    stdev = 0
    if len(data) > 1:
        for element in data:
            stdev += pow(element - mean, 2)
        stdev = stdev / (len(data) - 1)
    return pow(stdev, .5)

def calc_average_results(results):
    num_points = len(results[0])
    num_trials = len(results)
    avg_results = []
    stdev_results = []

    for point in range (0, num_points):
        column = []
        point_avg = 0.0
        for trial in range (0, num_trials):
            point_avg += results[trial][point]
            column.append(results[trial][point])
        
        mean = point_avg/num_trials
        stdev = calc_stdev(column, mean)                    
        avg_results.append(round(mean, 3))
        stdev_results.append(round(stdev, 3))
                          
    results.append(stdev_results)
    results.append(avg_results)

    return results


def u_label(u_labeled, u_unlabeled, m, sigma, k):
    m_uncertainties = calc_uncertainties(u_labeled, u_unlabeled, sigma, k)[:m]
    for to_label in m_uncertainties:
         u_labeled.append(to_label)
         u_unlabeled.remove(to_label)

def r_label(r_labeled, r_unlabeled, m):
    for i in range (0, m):
        if r_unlabeled:
            #random.seed(10000)
            index = random.randint(0, len(r_unlabeled)-1)
            r_labeled.append(r_unlabeled.pop(index))
        
    
def calc_num_classes(dataset):
    classes = []
    for member in dataset:
        if member['class'] not in classes:
            classes.append(member['class'])

    return len(classes)

def write_files(u_results, r_results, num_labeled, k, u_filep, r_filep):
    u_file = open(u_filep, 'w')
    r_file = open(r_filep, 'w')
    
    header = "K = " + str(k) + "\n\nNumber of labeled instances\n"
    for ins in num_labeled:
        header += str(ins) + ","
    header += "\n\nValidation Set Accuracy\n"

    u_file.write(header)
    r_file.write(header)

    text = ""
    for trial in range (0, len(u_results)):
        text += "Trial " + str(trial) + ": "
        for point in u_results[trial]:
            text+= str(point) + ","
        text += "\n"
    u_file.write(text)
            
    text = ""
    for trial in range (0, len(r_results)):
        header += "Trial " + str(trial) + ": "
        for point in r_results[trial]:
            header += str(point) + ","
    r_file.write(text)

def make_plot(u_results, r_results, num_labeled, m):
    x = num_labeled
    yu = u_results[len(u_results)-1]
    yr = r_results[len(r_results)-1]
    yuerr = u_results[len(u_results)-2]
    yrerr = r_results[len(r_results)-2]
    
    u_plot = plt.figure()
    uax = u_plot.add_subplot(111)
    uax.set_title('Accuracy of Uncertainty Sampling')
    uax.set_ylabel('Accuracy')
    uax.set_xlabel('Labeled Instances')
    uax.errorbar(x, yu, yerr=yuerr, fmt='b')
    uax.errorbar(x, yr, yerr=yrerr, fmt='r')
    uax.axis([0, num_labeled[len(num_labeled)-1]+m, 0, 1])
   
    plt.show()


def main(args):
    if args[3] == "ecoli.arff":
        sigma = .75
        v = 70
    elif args[3] == "ionosphere.arff":
        sigma = 3.0
        v = 40
    else:
        print "unknown input file"
        sigma = 1
        v = 10


    k = int(args[2])
    m = 5
    data = get_training(args[3])
    u_results = []
    r_results = []   
    num_labeled = []
    
    for i in range (0, 10):
        # print "Trial " + str(i)
        temp_data = copy.deepcopy(data)
        validation = get_validation(temp_data, v)
        u_labeled = get_labeled(temp_data, k)
        r_labeled = copy.deepcopy(u_labeled)
        u_unlabeled = temp_data
        r_unlabeled = copy.deepcopy(u_unlabeled)
        trial_u_results = []
        trial_r_results = []
        u_accuracy = 0
        r_accuracy = 0
       
        # calculate accuracy of initial set
        if i == 0:
            num_labeled.append(len(r_labeled))
        
        u_accuracy = validate(u_labeled, validation, k, sigma)
        trial_u_results.append(round(u_accuracy, 3))
        
        r_accuracy = validate(r_labeled, validation, k, sigma)
        trial_r_results.append(round(r_accuracy, 3))
        
        while u_unlabeled:
            # uncertainty sample
            u_label(u_labeled, u_unlabeled, m, sigma, k)
            u_accuracy = validate(u_labeled, validation, k, sigma)
            trial_u_results.append(round(u_accuracy, 3))
        
            
            # random sample
            r_label(r_labeled, r_unlabeled, m)
            r_accuracy = validate(r_labeled, validation, k, sigma)
            trial_r_results.append(round(r_accuracy, 3))
            
            # make list of number of labeled elements
            if i == 0:
                num_labeled.append(len(u_labeled))
            
        u_results.append(trial_u_results)
        r_results.append(trial_r_results)
           
    # calculate average of all trials
    r_results = calc_average_results(r_results)
    u_results = calc_average_results(u_results)
    
    # add case for 0 labeled instances
    # r_results[len(r_results)-1].insert(0, 0)    
    # u_results[len(u_results)-1].insert(0, 0)
    # r_results[len(r_results)-2].insert(0, 0)    
    # u_results[len(u_results)-2].insert(0, 0)
    # num_labeled.insert(0, 0)

    # write files
    write_files(u_results[:-2], r_results[:-2], num_labeled, k, args[5],args[4])

    # make plot
    # make_plot(u_results, r_results, num_labeled, m)


if __name__ == "__main__":
    main(sys.argv)
