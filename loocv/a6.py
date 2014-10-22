from operator import itemgetter
import math, sys, knn, time, zscore, selection

def main(args):
    # FILTER PART A
    print "Filter Method:\n"
    training_data = knn.get_training(args[1])    
    k = 7
    correlations = selection.apply_pearson(training_data)

    print str("Part A: Features listed in descending order"
              " according to the |r| value")
    for feature, value in correlations:
        print "Feature " + str(feature) + " has an |r| of " + str(value)
    print ""

    # FILTER PART B
    print "Part B: Values of m and Avg LOOCV accuracy"
    selection.filter_method(correlations, training_data, k)   


    # WRAPPER 
    training_data = knn.get_training(args[1])
    k = 7
    print "\n\nWrapper Method:\n"
    print "Iteration number, selected features and avg LOOCV accuracy"
    selection.wrapper_method(training_data, k)


    # CUSTOM
    print "\n\nCustom Method:\n"
    training_data = knn.get_training(args[1])    
    k = 7
    correlations = selection.apply_pearson(training_data)
    stdev_mean = zscore.get_stdev_mean(training_data)
    selection.custom_method(correlations, training_data, k, stdev_mean)

if __name__ == "__main__":
    main(sys.argv)
