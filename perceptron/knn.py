#!/usr/bin/python
'''
COMP 135, Spring 2014
'''
import sys, copy, arff, string, zscore

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
   # print args
    normalize = args[2]
    ks = []
    ks.append(int(args[4]))
    
    training = get_training(args[5])
   # print training
    output = open(args[7], 'w')
    test = open(args[6], 'r')
    if normalize == '1':
        stdev_mean = zscore.get_stdev_mean(training)
        training = zscore.normalize_training(training, stdev_mean)

    # number of classifications that differ from the known class
    num_error = 0.0

    # number of total data points
    num_points = 0.0
        
    # tree built from training data 
    t = kdtree(training)

    # accumulating string of output
    text = ""
    
    dimensions = len(training[0]['point'])
    for line in test :
        attrs = string.split(line,',')
        if len(attrs) == dimensions + 1 : #point to test 
            num_points += 1
            
            #get floats for this data pointfd
            point = map(float, attrs[:-1])
            if normalize == '1': #normalize test data
                point = zscore.normalize_test(point, stdev_mean)
                #print point[0]
                #sys.exit()

            #prepare knn search for this data point
            knn = lambda k : chooseBest(getNeighbors(dimensions,t,point,k,[]))
            line = []
            line.extend(attrs[:-1]) #data point
            line.extend(map(knn,ks)) #knn solutions
            if attrs[len(attrs) - 1] != line[len(line) - 1] + '\n':
                num_error +=1
            
            #append text with commas
            text += string.join(line,',') + '\n'
            
        else : #header
            text += line 
   # print round(100 * (1 - (num_error / num_points)))
    output.write(text)

    test.close()
    output.close()


    
def avg_h(neighbors):
    # pulls out average of all sqdists
    accum = 0
    for neighbor in neighbors :
        accum += neighbor['sqdist']
    return accum / len(neighbors)

def chooseBest(neighbors):
    i = 0
    classes = {}
    # build dictionary : classes are keys, values are lists
    # of relevant neighbors 
    for neighbor in neighbors :
        class_name = neighbor['class']
        if not class_name in classes :
            classes[class_name] = [neighbor]
        else :
            classes[class_name].append(neighbor)
    
    # find key value pair with best length of relative neighbors
    # manage ties with average closeness
    bestavg = float('inf')
    bestcount = 0
    for class_name, neighbors in classes.iteritems():
        if len(neighbors) > bestcount :
            bestclass = class_name
            bestcount = len(neighbors)
            bestavg = avg_h(neighbors) 
        elif len(neighbors) == bestcount:
            accum = 0
            avg = avg_h(neighbors) 
            if avg < bestavg :
                bestavg = avg
                bestclass = class_name
    return bestclass

def kdtree(pointlist):
    k = len(pointlist[0]['point'])
    # initial bounds start at -inf -> +inf on all axes
    bounds = [{'min':-float('inf'),'max':float('inf')} for axis in range(k)]
    return kdtree_assist(k, pointlist, bounds, 0)

def kdtree_assist(k, pointlist, bounds, depth):
    if len(pointlist) == 0 : 
        # empty obj as leaves
        return {}

    axis = depth % k; # rotate through axis as descending through tree

    # sort points, split on median
    pl = sorted(pointlist, key=lambda x: x['point'][axis])
    if len(pl) == 1 : 
        medianIndex = 0
    else :
        medianIndex = (len(pl) + 1) / 2;
    median = pl[medianIndex]
    left   = pl[:medianIndex]
    right  = pl[medianIndex + 1:]

    # generate boundary regions for children
    l_bounds = copy.deepcopy(bounds)
    l_bounds[axis]['max'] = median['point'][axis] 
    r_bounds = copy.deepcopy(bounds)
    r_bounds[axis]['min'] = median['point'][axis] 

    # build node of tree around the median
    node = {
        'axis'      : axis,
        'point'     : median['point'],
        'class'     : median['class'],
        'l'         : kdtree_assist(k,left, l_bounds, depth + 1),
        'r'         : kdtree_assist(k,right,r_bounds, depth + 1),
        'bounds'    : bounds
    }

    return node


def getNeighbors(k,tree, point,num, neighbors):
    # contract, neighbors is sorted by furthest to closest
    if not tree : 
        return
    rsearched = False
    lsearched = False
    if not neighbors : #no elements in list
        if tree['r'] and point[tree['axis']] > tree['point'][tree['axis']]:
            rsearched = True
            neighbors = getNeighbors(k,tree['r'],point,num,neighbors)
        elif tree['l'] and point[tree['axis']] > tree['point'][tree['axis']] :
            lsearched = True
            neighbors = getNeighbors(k, tree['l'], point,num, neighbors)

    # check self for better neighborhood if within hypersphere
    # neighbors only added to list in this case
    if len(neighbors) < num or neighbors[0]['sqdist'] >=  eu_sq(k, point, tree['point']) :
        addneighbor(num, neighbors, {
            'point' : tree['point'],
            'class' : tree['class'],
            'sqdist'   : eu_sq(k, point, tree['point'])
        })

    # search right subtree
    if not rsearched and tree['r'] and \
            neighbors[0]['sqdist'] >= region_dist(tree['r'], point) :
        neighbors = getNeighbors(k, tree['r'],point, num, neighbors)

    # search left subtree
    if not lsearched and tree['l'] and \
            neighbors[0]['sqdist'] >= region_dist(tree['l'], point) :
        neighbors = getNeighbors(k, tree['l'],point, num, neighbors)
    return neighbors

def addneighbor(num, neighbors, newneighbor):
    # adds neighbor to neighbors list
    # keeps length of neighbors less than or equal to num unless there's a tie
    # keeps neighbors sorted largest to smallest distance from considered point 
    count = len(neighbors)
    i = 0
    while i < count and neighbors[i]['sqdist'] > newneighbor['sqdist']:
        i += 1
    
    neighbors.insert(i,newneighbor)
    count += 1

    if count <= num :
        return neighbors

    i = 1
    while i < count and neighbors[i-1]['sqdist'] == neighbors[i]['sqdist']:
        # count the number tied at the front of the list
        i += 1
    
    if count - i >= num :
        # remove tied if they alone bring the count above num
        while i > 0 :
            neighbors.pop(0)
            i -= 1
    return neighbors

def eu_sq(k,p1,p2):
    # returns euclidian distance squared
    d = 0
    for dimension in range(k):
        d += (p1[dimension] - p2[dimension])**2
    return d


def region_dist(region, point):
    # distance between a point and the closest point in the boundary region
    bounds = region['bounds']
    closest_point = []
    for dimension in range(len(bounds)):
        if point[dimension] < bounds[dimension]['min'] :
            closest_point.append(bounds[dimension]['min'])
        elif point[dimension] > bounds[dimension]['max'] :
            closest_point.append(bounds[dimension]['max'])
        else :
            closest_point.append(point[dimension])
    return eu_sq(len(bounds), point, closest_point)


if __name__ == "__main__":
    main(sys.argv)
