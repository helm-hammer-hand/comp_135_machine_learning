import sys, random, knn, zscore, string

def build_perceptron(eta, training_data):
    #add constant 1.0 x0 for calculation of w0
    for example in training_data:
        example['point'].insert(0, 1.0)

    #initialize weight and delta weight vectors
    num_features = len(training_data[0]['point']) #num real features + 1
    weights = [0.0] * num_features
    
    #initialize output vector
    num_examples = len(training_data)
    for i in range (0, 500):
        for ex in range (0, num_examples):
            o = compute_output(training_data[ex], weights)
            for wi in range (0, num_features):
                t = float(training_data[ex]['class'])
                xi = training_data[ex]['point'][wi]
                delta = eta * (t - o) * xi
                weights[wi] += delta
    return weights

def compute_output(example,weights):
    out = 0.0
    for feature in range (0, len(weights)):
        out += example['point'][feature] * weights[feature]
    return out

def apply_perceptron(test_data, weights):
    calc_class = 0
    curr_ex = 0
    num_error = 0.0
    for example in test_data:
        output = weights[0]
        for feature in range (0, len(weights) - 1):
            output += weights[feature+1] * test_data[curr_ex]['point'][feature]
        if output > 0:
            calc_class = 1
        else:
            calc_class = -1
        if calc_class != int(test_data[curr_ex]['class']):
            num_error += 1
      
        test_data[curr_ex]['class'] = calc_class
        curr_ex += 1
    #print round(100 * (1.0 - (num_error / curr_ex)))
    return test_data
    
def write_file(test_data, test_file, out_file):
    test_orig = open(test_file, 'r')
    out = ""
    for line in test_orig:
        if len(string.split(line)) != len(test_data[0]['point']):
            out += line
    for example in test_data:
        for feature in example['point']:
            out += str(feature) + ","
        out += str(example['class']) + "\n"
    output = open(out_file, 'w')
    output.write(out)

def main(args):
    eta = float(args[2])
    training = knn.get_training(args[3])
    test = knn.get_training(args[4])

    #normalize data
    stdev_mean = zscore.get_stdev_mean(training)
    training = zscore.normalize_training(training, stdev_mean)
    test = zscore.normalize_training(test, stdev_mean)

    weights = build_perceptron(eta, training)
    test = apply_perceptron(test, weights)
    
    write_file(test, args[4], args[5])
    



if __name__ == "__main__":
    main(sys.argv)
