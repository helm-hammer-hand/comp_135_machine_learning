from operator import itemgetter
import math, sys, arff, zscore, time, copy, kmeans

def init_clusters_custom(k, data):
    clusters = []
    num_examples = len(data)
    offset = num_examples / k -1
   # print num_examples, offset
    for init in range (0, k):
        temp_dict = {}
        #print offset*init
        temp_c = copy.deepcopy(data[offset*init]['point'])
        temp_dict['centroid'] = temp_c
        temp_dict['members'] = []
        clusters.append(temp_dict)
        

    num_features = len(data[0]['point'])
    for example in data:
        min_dist = float("inf")
        min_cluster = -1
        for cluster in range (0, k):
            dist = kmeans.calc_dist(example['point'],
                                    clusters[cluster]['centroid'])
            if dist < min_dist:
                min_dist = dist
                min_cluster = cluster
        clusters[min_cluster]['members'].append(example['point'])
    #print clusters[0]
    return clusters


def kmeans_custom(k, data, stdev_mean):
    num_examples = len(data)
    num_features = len(data[0]['point'])
    min_sse = float("inf")
    min_sse_clusters = []
    sse_list = []
    
    clusters = init_clusters_custom(k, data)
    # do 50 iterations
    for i in range (0, 49):
        kmeans.calc_cluster_centroids(clusters)
        if kmeans.reassign_clusters(clusters) == False:
          #  print "break at " + str(i)
            break
      
    sse = kmeans.calc_sse(clusters)
    if sse < min_sse:
        min_sse_clusters = copy.deepcopy(clusters)
        min_sse = sse
   
    print "k = " + str(k)
    kmeans.print_cluster_centroids(min_sse_clusters, stdev_mean)
    print ""

def main(args):
    data = kmeans.get_data(args[1])
    stdev_mean = zscore.get_stdev_mean(data)
    z_data = zscore.normalize_training(data, stdev_mean)
    data = kmeans.get_data(args[1])
    
    for k in range (1, 17):
        kmeans_custom(k, z_data, stdev_mean)
                                   

if __name__ == "__main__":
    main(sys.argv)
