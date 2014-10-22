import sys, random

def main(args):
    num_clusters = int(args[2])
    num_features = int(args[4])
    min_size = int(args[6])
    max_size = int(args[8])
    outlier_percent = float(args[10])

    data = []

    index = 0
    feature_vals = []
    for i in range (1, 5*num_features + 1, 5):
        feature_vals.append(i)

    # build clusters
    for c in range (0, num_clusters):
        size = random.randint(min_size, max_size + 1)
        for ex in range (0, size):
            temp = {}
            temp['index'] = index
            temp['class'] = c
            temp['point'] = []
            current_index = c
     
            for f in range (0, num_features):
                offset = random.uniform(-1, 1)
                temp['point'].append(round(feature_vals[current_index] + offset,3))
        
                current_index += 1
                if current_index >= num_features:
                    current_index = 0
            
            data.append(temp)
            index += 1

    # build outliers
    num_outliers = int(outlier_percent * index)
    for i in range (0, num_outliers):
        temp = {}
        temp['index'] = index + i
        temp['class'] = -1  # to differentiate outliers
        temp['point'] = []
        
        #ind = random.randint(0, num_features - 1)
        ind = random.randint(0, index - 2)
        for j in range (0, num_features):
            #temp['point'].append(feature_vals[ind])
            d = random.sample(set([5, .2]), 1)[0]
            temp['point'].append(data[ind]['point'][j] / d)
        
        data.append(temp)
        
    write_file(args[11], data)

    
def write_file(filepath, data):
    f = open(filepath, 'w')

    text = "@relation synthetic\n"
    for i in range (0, len(data[0]['point'])):
        text += "@attribute F" + str(i+1) + " numeric\n"
    text += "@attribute class numeric\n" + "@data\n"

    for ex in data:
        for feat in range (0, len(ex['point'])):
            text += str(ex['point'][feat]) + ","
        text += str(ex['class']) + "\n"
                 
    f.write(text)


if __name__ == "__main__":
    main(sys.argv)
