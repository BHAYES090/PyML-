from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random 

style.use('fivethirtyeight')

#assigning data below

#xs = np.array([1,2,3,4,5,6], dtype=np.float64) 
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

#creating a dataset of random numbers to test

def create_dataset(hm, variance, step = 2, correlation = False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

#assigning the function for y = mx + b (which is different from the traditional)
#formula for a slope of a line becuase this is calculating the best fit line
#from the data set.

#defining the best fit line

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
          ((mean(xs) * mean(xs)) - mean(xs ** 2)))
    b = mean(ys) - m*mean(xs)
    return m, b

#creating new dataset
#correlation and variance can be adjusted to determine how accuratly the line
#falls, see def create_dataset
xs, ys = create_dataset(40, 10, 2, correlation = 'pos')

# definind squared error

#how good is the best fit line to this data? To calculate how accurate a best
#fit line is I will calculate squared error which is the distance between a
#point and the line squared, depending on how much you want to "penalize" the
#line depending on how scattered your data points are from the line. it can be
#error (e) to the 4,6, or 18 power depending on how much you want to correct
#your line.

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

#def coefficient of determination

#r squared is the coefficient of determination it is equal to the squared error
#(SE) * y hat or (the regression line or best fit line) all devided by the
#squared error times the mean of the ys r squared value should be a higher
#number (or as high as possible)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


m,b  = best_fit_slope_and_intercept(xs,ys)

regression_line = [(m * x) + b for x in xs]

#predicting where the best fit line will fall at the value of '8' on the graph
#prediction can be commented out to see just the plain line without prediction

predict_x = 8
predict_y = (m * predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color =('g'))
plt.plot(xs, regression_line)
plt.show()
