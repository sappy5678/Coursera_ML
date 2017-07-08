import pandas as pd
from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt

# csv = np.genfromtxt ('kc_house_train_data.csv', delimiter=",")
# second = csv[:,1]
# third = csv[:,2]
train_data = pd.read_csv("kc_house_train_data.csv")
test_date = pd.read_csv("kc_house_test_data.csv")
test_date_11x = pd.read_csv("1+1x.csv")
def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    input_sum = input_feature.sum()
    output_sum = output.sum()

    # compute the product of the output and the input_feature and its sum
    sum_of_input_mul_output = (input_feature * output).sum()
    # compute the squared value of the input_feature and its sum
    sum_of_input_mul_input = (input_feature * input_feature).sum()
    # use the formula for the slope(w1)
    slope = (sum_of_input_mul_output - input_sum * output_sum  / input_feature.count()) / \
            (sum_of_input_mul_input - input_sum * input_sum / input_feature.count())
    # use the formula for the intercept(w0)
    intercept = output.mean() - slope * input_feature.mean()
    return (intercept, slope)


(test_intercept, test_slope) =  simple_linear_regression(test_date_11x["x"], test_date_11x["y"])
print "Intercept: " + str(test_intercept)
print "Slope: " + str(test_slope)


