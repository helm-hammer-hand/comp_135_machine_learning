from operator import itemgetter
import math, sys, arff, zscore, time, copy

# this is from the solution to hw1
def get_data(data_file):
    training = []
    for row in arff.load(data_file) :
        row = list(row)
        training.append({
            #'class' : row.pop(),
            'point' : row
        })
    return training

def get_instances(data_file):
    instances = []
    f = open(data_file, 'r')
    for line in f:
        instances.append(int(line))
    return instances

def calc_dist(point_a, point_b):
    """ takes in the 'point' entry of the example dictionary """
    num_features = len(point_a)
    dist = 0
    for feature in range (0, num_features):
        dist += pow(point_a[feature] - point_b[feature], 2)
    return pow(dist, .5)

def calc_cluster_centroids(clusters):
    num_features = len(clusters[0]['centroid'])
    
    for cluster in clusters:
        cluster_size = len(cluster['members'])
        for feature in range (0, num_features):
            avg = 0.0
            for member in cluster['members']:
                avg += member[feature]
            avg = avg / cluster_size
            cluster['centroid'][feature] = avg

def calc_sse(clusters):
    num_features = len(clusters[0]['members'][0])
    sse = 0.0
    for cluster in clusters:
        for member in cluster['members']:
            temp = calc_dist(member, cluster['centroid'])
            sse += pow(temp, 2)
    return sse

                
def print_cluster_centroids(clusters, stdev_mean):
    i = 0
    for cluster in clusters:
        for feature in range (0, len(cluster['centroid'])):
            mean = stdev_mean[feature][0]
            stdev = stdev_mean[feature][1]
            value = zscore.un_zscore(cluster['centroid'][feature], stdev, mean)
            sys.stdout.write(str(value) + ',')
        print ""
        i += 1

def reassign_clusters(clusters):
    num_clusters = len(clusters)
    reassigned = False
    
    new_clusters = copy.deepcopy(clusters)

    for nc in new_clusters:
        nc['members'] = []
    for current_c in range(0, num_clusters):
        for element in clusters[current_c]['members']:
            min_dist = float("inf")
            min_cluster = -1
           
            for new_c in range(0, num_clusters):
                dist = calc_dist(element, clusters[new_c]['centroid'])
                if dist < min_dist:
                    min_dist = dist
                    min_cluster = new_c
            if min_cluster != current_c:
                reassigned = True
            new_clusters[min_cluster]['members'].append(element)
    clusters = new_clusters             
    return reassigned

def init_clusters_random(k, data, instances, start_index):
    clusters = []
    instance_index = start_index
    for cluster in range (0, k):
        temp_dict = {}
        temp_c = copy.deepcopy(data[instances[instance_index]]['point'])
        temp_dict['centroid'] = temp_c
        clusters.append(temp_dict)
        clusters[cluster]['members'] =  []
        instance_index += 1

    #fill them
    num_features = len(data[0]['point'])
    for example in data:
        min_dist = float("inf")
        min_cluster = -1
        for cluster in range (0, k):
            dist = calc_dist(example['point'], clusters[cluster]['centroid'])
            if dist < min_dist:
                min_dist = dist
                min_cluster = cluster
        clusters[min_cluster]['members'].append(example['point'])
   
    return clusters


def kmeans(k, data, instances, stdev_mean):
    num_examples = len(data)
    num_features = len(data[0]['point'])
    min_sse = float("inf")
    min_sse_clusters = []
    sse_list = []
    min_attempt = 0

    for attempt in range (0, 25): 
       # print attempt
        #choose initial clusters
        start_index = k * attempt
        clusters = init_clusters_random(k, data, instances, start_index)
      #  for cluster in clusters:
      #      print cluster['centroid']
      #      print ""
        
        calc_cluster_centroids(clusters)
        
        # do 50 iterations
        for i in range (0, 49):
            if reassign_clusters(clusters) == False:
            #    print "broke at " + str(i)
                break
            calc_cluster_centroids(clusters)
   
        sse = calc_sse(clusters)
   #     print sse
        sse_list.append(sse)
        if sse < min_sse:
            min_attempt = attempt
            min_sse_clusters = copy.deepcopy(clusters)
            min_sse = sse
    #print "try number: " + str(min_attempt)
    sse_avg = zscore.calc_mean(sse_list)
    sse_stdev = zscore.calc_stdev(sse_list, sse_avg)
    print "k = " + str(k)
    print_cluster_centroids(min_sse_clusters, stdev_mean)
    print ""

def main(args):
    data = get_data(args[1])
    stdev_mean = zscore.get_stdev_mean(data)
    z_data = zscore.normalize_training(data, stdev_mean)
    data = get_data(args[1])
 
    instances = get_instances('instances.txt')
    
    for k in range (1, 17):
        kmeans(k, z_data, instances, stdev_mean)
                                   

if __name__ == "__main__":
    main(sys.argv)
