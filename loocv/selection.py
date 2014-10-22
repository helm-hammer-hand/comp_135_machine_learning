from operator import itemgetter
import math, sys, knn, time, zscore

def pearson(x, y):
    sum_sq_x = 0
    sum_sq_y = 0
    sum_coproduct = 0
    mean_x = 0
    mean_y = 0
    N = len(x)
    
    for i in range (0, N):
        sum_sq_x += x[i] * x[i]
        sum_sq_y += y[i] * y[i]
        sum_coproduct += x[i] * y[i]
        mean_x += x[i]
        mean_y += y[i]

    mean_x = mean_x / N
    mean_y = mean_y / N
    pop_sd_x = math.sqrt((sum_sq_x/N) - (mean_x * mean_x))
    pop_sd_y = math.sqrt((sum_sq_y / N) - (mean_y * mean_y))
    cov_x_y = (sum_coproduct / N) - (mean_x * mean_y)
    correlation = cov_x_y / (pop_sd_x * pop_sd_y)

    return correlation

def apply_pearson(training_data):
    correlations = {}
    stdevs = []
    for feature in range (0, len(training_data[0]['point'])):
        
        feat_array = []
        class_array = []
        for example in range (0, len(training_data)):
            feat_array.append(training_data[example]['point'][feature])
            class_array.append(float(training_data[example]['class']))
            
        r = pearson(feat_array, class_array)
        correlations[feature] = abs(r)

    corr_list = sorted(correlations.items(), key=itemgetter(1), reverse=True)
    
    return corr_list

def custom_method(correlations, training_data, k, stdev_mean):
    num_features = len(correlations)

    sorted_features = []
    current_features = []
    
    # append feature numbers to stdev_mean
    feat_num = 0
    for elem in stdev_mean:
        elem.append(feat_num)
        feat_num += 1
    
    for feature in correlations:
        new = []
        new.append(feature[0])
        new.append(feature[1] * (1 / stdev_mean[feature[0]][1]))
        sorted_features.append(new)
        
    sorted_features.sort(key=itemgetter(1), reverse=True)
    for feature in sorted_features:
        print "Feature " + str(feature[0]) + " has a weight of " \
            + str(feature[1])
    print ""
    
    print "Values of m and Avg LOOCV accuracy"
    for m in range (0, num_features):
        current_features.append(sorted_features[m][0])
        ans = loocv(training_data, k, current_features)
        print "M: " + str(m+1) + ", LOOCV Accuracy: " + str(ans[0]) + "/" \
            + str(ans[1]) + ", " + str(round(ans[2], 1)) + \
            "% Correctly Classified"

def filter_method(correlations, training_data, k):
    num_features = len(correlations)

    current_features = []
    for m in range (0, num_features):
        current_features.append(correlations[m][0])
        ans = loocv(training_data, k, current_features)
        print "M: " + str(m+1) + ", LOOCV Accuracy: " + str(ans[0]) + "/" \
            + str(ans[1]) + ", " + str(round(ans[2], 1)) + \
            "% Correctly Classified"

def wrapper_method(training_data, k):
    num_features = len(training_data[0]['point'])
    num_examples = len(training_data)
    current_features = []
    
    total_correct = 0
    temp_correct = 0
    i = 0
    for test in range (0, num_features):
        print_line(num_examples, total_correct, i, current_features)
        temp_correct = -1
        best_feature = -1
        for feature in range (0, num_features):
            if feature not in current_features:
                current_features.append(feature)
                feat_correct = loocv(training_data, k,
                                     current_features)[0]
           
                if feat_correct > temp_correct:
                    best_feature = feature
                    temp_correct = feat_correct
            
                current_features.pop()

        if temp_correct > total_correct:
            current_features.append(best_feature)
            total_correct = temp_correct
        else:
            break
            
        i += 1

def print_line(num_examples, num_correct, i, current_features):
    sys.stdout.write("Iteration " + str(i) + ", Selected Features: { ")
    for feat in current_features:
        sys.stdout.write(str(feat) + " ")
    sys.stdout.write("} LOOCV Accuracy: " + str(num_correct) + "/" + \
                     str(num_examples) + ", " + \
                     str(round(100.0 * num_correct / num_examples, 1)) + \
                     "% Correct\n")

def loocv(training_data, k, current_features): 
    num_examples = len(training_data)
    num_correct = 0
    for leave_out in range (0, num_examples):
        test = [training_data.pop(leave_out),]
        num_correct += knn.knn(training_data, test, k, current_features)
        training_data.insert(leave_out, test[0])
    acc = 100.0 * num_correct / num_examples
    return [num_correct, num_examples, acc]

