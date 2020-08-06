import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = pd.read_csv(r'C:\Users\bohayes\AppData\Local\Programs\Python\Python38\forestfires.csv')
df = pd.DataFrame(data, columns= ['X', 'Y'])

##X = pd.DataFrame(data, columns = ['X'])

##Y = pd.DataFrame(data, columns = ['Y'])

##Z = np.tan(X)
X = [-1, 4, -0.6]
Y = [8, 34, -500]

ReLU=[]

ReLU1 = []

def linear_kernel(X, Y):
    Z = X*Y
    return Z

for i in linear_kernel(X, Y):
    ReLU.append(max(0, i))


for i in linear_kernel(X, Y):
    if i <= 0:
        ReLU1.append(0)

print(ReLU)
print(ReLU1)

##def linear_kernel(X, Y):
##    return np.dot((X), np.array(Y).T)
##
##for i in np.nditer(linear_kernel(X, Y)):
##    ReLU.append(max(0, i))
##
##print(ReLU)

##V = np.nditer(linear_kernel(X, Y), op_flags=['readwrite'])
##while not V.finished:
##    if V 0:
##        print(0)

##for x in np.nditer(linear_kernel(X, Y), op_flags=['readwrite']):
##    if x <= 0:
##        
##    print(x)
##    np.column_stack(np.nditer(linear_kernel(X, Y)))
##    print(np.column_stack)
##    print(x, end=' ')

####def ReLU(X, Y):
##    for x in np.nditer(linear_kernel(X, Y), op_flags=['readwrite']):
##        print(x, end=' ')
##        if x > 0:
##            return x
##        if x <= 0:
##            return 0

##print(Z)
##print(linear_kernel(X, Y))
##print(np.arange(ReLU(X, Y)))
##ax = plt.axes(projection='3d')
##ax.scatter3D(X, Y, Z)
##ax.axes

##for cell in np.nditer(linear_kernel(X, Y)):
##    print(cell, end=' ')

##plt.scatter(Y, Z, color='r')
##plt.scatter(X, Y)

##plt.show()
