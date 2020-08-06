import numpy as np
from math import e
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#######################LAYER 1############################

inputs0 = [0.9, 0.8774, 0.2323, 0.71]
weights0 = [-0.001, 0.12224, -0.221, -0.2322]

inputs1 = [0.897, 0.665, -0.112, -0.645]
weights1 = [-0.1455, -0.221, 0.999, -0.00122]

inputs2 = [0.232, -0.115, 0.712, 0.52]
weights2 = [0.444, -0.62, 0.112, -0.834]
biases0 = [0.1, 0.4, 0.5, 0.2]

output0 = np.dot(inputs0, weights0)
output1 = np.dot(inputs1, weights1)
output2 = np.dot(inputs2, weights2)

A = output0 + output1 + output2 + biases0

########################LAYER 2###########################

inputs3 = [A]
weights3 = [[0.0303, -0.523, 0.756],
            [0.6, -0.075, 0.43],
            [0.23, -0.04, 0.57],
            [0.0112, -0.42, 0.5]]

inputs4 = [A]
weights4 = [[0.33, -0.0234, 0.675],
            [0.5, -0.57, 0.034],
            [0.0321, -0.03, 0.75],
            [0.121, -0.0204, 0.4]]

inputs5 = [A]
weights5 = [[0.0312, -0.342, 0.0743],
            [0.3, -0.078, 0.122],
            [0.0213, -0.2, 0.059],
            [0.144, -0.0265, 0.3]]
biases1 = [0.3, 0.4, 0.5]


output3 = np.dot(inputs3, weights3)
output4 = np.dot(inputs4, weights4)
output5 = np.dot(inputs5, weights5)

B = output3 + output4 + output5 + biases1

########################LAYER 3#########################

inputs6 = [B]
weights6 = [[0.02, -0.03, 0.02],
            [0.03, -0.04, 0.03],
            [00.04, -0.05, 0.04]]

inputs7 = [B]
weights7 = [[0.02, -0.03, 0.02],
            [0.03, -0.04, 0.03],
            [0.04, -0.05, 0.04]]


inputs8 = [B]
weights8 = [[0.02, -0.03, 0.020],
            [0.03, -0.40, 0.03],
            [0.04, -0.05, 0.04]]
biases2 = [0.012, 0.034, 0.05]

output6 = np.dot(inputs6, weights6)
output7 = np.dot(inputs7, weights7)
output8 = np.dot(inputs8, weights8)

C = output6 + output7 + output8 + biases2

#########################LAYES 4#########################

inputs9 = [C]
weights9 = [[0.033232, -0.025675],
            [0.03543245, 0.03123],
            [-0.0445, 0.05455]]

inputs10 = [C]
weights10 = [[0.02456, 0.024545]
             ,
            [0.044545, -0.034564],
            [0.044564, 0.0445645]]

inputs11 = [C]
weights11 = [[-0.02786, 0.03453],
            [0.030782, 0.034561],
            [0.50452, -0.0412354]]
biases3 = [0.021, 0.403]

output9 = np.dot(inputs9, weights9)
output10 = np.dot(inputs10, weights10)
output11 = np.dot(inputs11, weights11)

Z = output9 + output10 + output11 + biases3

print(A, "outputs from Layer 1")
print(B, "outputs from Layer 2")
print(Z, 'Identity outputs from Layer 3') #Identity
print(np.log(1 + np.exp(Z)), "SoftPlus") #SoftPlus
print((1)/1 + np.exp(-Z), "Sigmoid") #Sigmoid
print(np.arctan(Z), "ArcTan") #ArcTan
print(np.tanh((2) / (1 + np.exp(-2*Z)-1)), "Tanh") #TanH

################################See Below for the beginning of a 3-d plot,
################################All numbers need to be moved one more palyer to a 2-d matrix in order to plot 
fig = plt.figure()
ax = fig.add_subplot(111)
x = (1.44799657, 0.39754059)
y = (2.03245152, 1.67830193)
plt.scatter(x, y, c='r', marker='o', linewidth=0.5)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.plot(x, y)
plt.show()

##############see WORKING sigmoid fuction as print((1)/1 + np.exp(-Z)) Derivitave of SoftPlus
##############see WORKING SoftPlus function as print(np.log(1 + np.exp(Z)))
##############see WORKING ArcTan function as print(np.arctan(Z))
##############see WORKING TanH function as print(np.tanh((2) / (1 + np.exp(-2*Z)-1)))
############
############print((1)/1 + np.exp(-Z))
##I turned all of the middle values of the weights to a negative and plugged the matracies
## into the sigmoid function... IDLE threw an error beacsue
##the values plugged into the sigmoid function are so infinately small that they cannot be accuratley caluclated

