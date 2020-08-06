import numpy as np
import matplotlib.pyplot as plt

# To return to original: add weights0 to the self.weights 

class Activation0:
    def __init__(self, X_0, weights0, biases0):
        self.X_0 = X_0
        self.weights0 = weights0
        self.biases0 = biases0
        self.layer_1 = np.dot(self.X_0, self.weights0) + self.biases0

    def ReluFunction0(self):
        ReLU = []
        for i in np.nditer(self.layer_1):
            ReLU.append(max(0, i))
        return ReLU

P1 = Activation0([[5.0, 0.11450, 451.00, 20.003],
        [0.89, 0.3322, 0.43232, 0.78899],
        [8.89844, 2.35447, 0.999999, 2.355],
        [12.4544, 2.3544, 12.245, 2311],
        [12.1211, 0.001212445, 23.5400, 0.11444],
        [0.111210, 13, 0.2323231, 0.03120012]],
                 [[0.225,-0.001],
                 [-0.32457, 0.445511],
                 [1.0, -0.99955],
                 [0.88774, -0.00023]],
                [-1.0, 1.0])

print(P1.ReluFunction0())
X_1 = P1.ReluFunction0()


#################################################################


##class Activation1:
##    def __init__(self, X_1, weights1, biases1):
##        self.X_1 = X_1
##        self.weights1 = weights1
##        self.biases1 = biases1
##        self.layer_2 = np.dot(self.X_1, self.weights1) + self.biases1
##
##    def ReluFunction1(self):
##        ReLU = []
##        for i in np.nditer(self.layer_2):
##            ReLU.append(max(0, i))
##        return ReLU
##
##P2 = Activation1(X_1,
##                [[-0.00121, 0.001121],
##                 [0.12121, 0.005588],
##                 [-0.12245, 0.0045478],
##                 [0.0056, -0.232211],
##                 [0.0001278, -0.0598898],
##                 [-0.12121, 0.000236],
##                 [-0.00121, 0.001121],
##                 [0.12121, 0.005588],
##                 [-0.12245, 0.0045478],
##                 [0.0056, -0.232211],
##                 [0.0001278, -0.0598898],
##                 [-0.12121, 0.000236]],
##                [-0.23, 0.23])
##print(P2.ReluFunction1())
##X_2 = P2.ReluFunction1()
##
##
##################################################################
##
##
##class Activation2:
##    def __init__(self, X_2, weights2, biases2):
##        self.X_2 = X_2
##        self.weights2 = weights2
##        self.biases2 = biases2
##        self.layer_3 = np.dot(self.X_2, np.array(self.weights2).T) + self.biases2
##
##    def ReluFunction2(self):
##        ReLU = []
##        for i in np.nditer(self.layer_3):
##            ReLU.append(max(0, i))
##        return ReLU
##
##P3 = Activation2(X_2,
##                 [[-0.75688, 0.889888],
##                  [-0.000232, 0.0778888],
##                  [-0.22556, -0.2366],
##                  [-0.01454111, 0.0565622],
##                  [-0.000556, 0.0078],
##                  [-0.1224441, 0.032388],
##                  [-0.00023, 0.004541],
##                  [-0.00450004, 0.002125421],
##                  [-0.0012144, 0.00288],
##                  [-0.56778, 0.00012487],
##                  [-0.04451, 0.4487],
##                  [0.05989, -0.56778]],
##                 [-0.2211, 0.22325, -0.89777, 0.5787, -0.00577, 0.54474,
##                  -0.45477, 0.8987, -0.145411, 0.2356411, 0.12445, 0.00232])
##
##print(P3.ReluFunction2())
##X_3 = P3.ReluFunction2()
##
##
#################################################################
##
##
##class Activation3:
##    def __init__(self, X_3, weights3, biases3):
##        self.X_3 = X_3
##        self.weights3 = weights3
##        self.biases3 = biases3
##        self.layer_4 = np.dot(self.X_3, np.array(self.weights3).T) + self.biases3
##
##    def ReluFunction3(self):
##        ReLU = []
##        for i in np.nditer(self.layer_4):
##            ReLU.append(max(0, i))
##        return ReLU
##
##P4 = Activation3(X_3,
##                 [[-0.000232, 0.0778888, -0.0238985, -0.232450014, 0.222562, -0.25200232,
##                   -0.000232, 0.0778888, -0.0238985, -0.232450014, 0.222562, -0.25200232],
##                  [-0.22556, -0.2366, 0.0112, 0.23555, 0.23224, 0.5565587, -0.1224441,
##                   0.032388, -0.56, -0.0002355612, 0.011200, 0.0232001]],
##                 [-0.2211, 0.22325])
##
##print(P4.ReluFunction3())
##X_4 = P4.ReluFunction3()
##
##
################################################################
##
##
##class Activation4:
##    def __init__(self, X_4, weights4, biases4):
##        self.X_4 = X_4
##        self.weights4 = weights4
##        self.biases4 = biases4
##        self.layer_5 = np.dot(self.X_4, np.array(self.weights4).T) + self.biases4
##
##    def ReluFunction4(self):
##        ReLU = []
##        for i in np.nditer(self.layer_5):
##            ReLU.append(max(0, i))
##        return ReLU
##
##P5 = Activation4(X_4,
##                 [[-0.1211, 0.068988]],
##                 [-0.045, 0.988])
##
##print(P5.ReluFunction4())
##X_5 = P5.ReluFunction4()
##
##
################################################################
##
##
##class Activation5:
##    def __init__(self, X_5, weights5, biases5):
##        self.X_5 = X_5
##        self.weights5 = weights5
##        self.biases5 = biases5
##        self.layer_6 = np.dot(self.X_5, self.weights5) + self.biases5
##
##    def ReluFunction5(self):
##        ReLU = []
##        for i in np.nditer(self.layer_6):
##            ReLU.append(max(0, i))
##        return ReLU
##
##P6 = Activation5(X_5,
##                 [[-0.12],
##                  [0.32]],
##                 [1.0])
##
##print(P6.ReluFunction5())
##X_6 = P6.ReluFunction5()
##
################################################################
##
##class Activation6:
##    def __init__(self, X_6, weights6, biases6):
##        self.X_6 = X_6
##        self.weights6 = weights6
##        self.biases6 = biases6
##        self.layer_7 = np.dot(self.X_6, self.weights6) + self.biases6
##
##    def ReluFunction6(self):
##        ReLU = []
##        for i in np.nditer(self.layer_7):
##            ReLU.append(max(0, i))
##        return ReLU
##
##P7 = Activation6(X_6,
##                 [[-0.23, 0.0778888, -0.0238985, -0.232450014, 0.222562, -0.25200232,
##                   -0.000232, 0.0778888, -0.0238985, -0.232450014, 0.222562, -0.25200232]],
##                 [-1, -0.2211, 0.22325, -0.232450014, 0.222562, -0.25200232,
##                  -0.000232, 0.0778888, -0.0238985, -0.232450014, 0.222562, -0.25200232])
##
##print(P7.ReluFunction6())
##
##
#################################################################
####
######Completed forward propogation minus the final round of selection and back propogation
####


