## Mchine Learning for Beginners

#Machine learning
#   it is simply the method of teaching computers to make
#   predictions based on some data provided as input
#   Over time, as more data is added as input, the machoine
#   learns the pattern and the output becomes more efficient 

# The different types of Machine learning
#    Reinforcement learning:
#             feedback to algorithm when it does something right or wrong
#             (mix of supervised and un-supervised)
#             Example: child gets feedback on the job when they do something right or wrong
#    Supervised learning:
#             pre-labelled data trains a model to predict new outcomes.
#             Dataset: has defined inputs and outputs
#             Objective: Train a moodel to predict outputs form future inputs
#             CLASSIFICATION: the output variable ois a category (Yes or No), or (Dog, cat, mouse)
#             REGRESSION: The output variable is a continuous value such as dollars, or credit risk
#             Example: Sorting LEGO blocks by matching them with the color of the bags
#    Unsupervised Learning:
#             non-labeled data self organises to predict new outcomes (Clusering)
#             Dataset: Only the inputs are known, the output is NOT known
#             Objective: Train a model to find existing patterns in the data to learn more about it.
#             ASSOCIATION: You want to discover rules that describe your data, such as people
#                          that buy beer also tend to buy diapers
#             CLUSTERING: You want to discover the inherent categories in the data, such as
#                         grouping customers by purchasing behavior

# Machine Learning Work Flow:
#    Data Collection:
#             For deep learning, we need a large quantity of accurate and consistent data
#    Data Preparation:
#             In this step, an analyst determines what parts of the data become inputs and outputs
#    Training:
#             In this step, ML engineers choose the best algorithm and iteratively tweak it while
#             comparing its predicted values to actual values to see how well it works
#    Inference:
#             If the object was for the model to make a prediction (e.g supervised learning),
#             then the model can be deployed so that it responds quickly to queries
#    feedback:
#             this is an optional step where information from the inferencing is used to update the
#             model so it accuracy can be improved

# Python tutorial

# Assignment of variable (string)
test = "Hello world"
print(test[0])
print(len(test))
print(test)

# assignment of Variable (Number)
value = 123.1
print(value)

# Boolean assignment
a = True
b = False
print(a, b)

# Multiple assignment
a, b, c = 1, 2, 3
print(a, b, c)

# None assignment
a = None
print(a)


##################################################################################################################


# Flow Control

# if-then-else conditional statement
value = 99

if value == 99:
    print('That is fast')
elif value > 200:
    print('That is TOO fast')
else:
    print('That is safe')

# For loop
for i in range(10):
    print(i)

# While loop
i = 0
while i < 10:
    print(i)
    i += 1


##################################################################################################################


# Functions 

# The example below defines a new function to calculate the sum of two values and
# calls the function with two arguments (Ensure that you have an empty line fter indented code,
# called white space)

# Sum Function
def sum(x, y):
    return x + y

#Test sum function
result = sum(1, 3)
print(result)

################################################################################################################


# Data Structures
# there are three data structures in python that will be the most useful,
# they are Tuples, lists and dictionaries

# Tuples
# tuples are a read-only collection of items

a = (1, 2, 3)
print(a)

# Lists
# Lists use the square bracket notation and can be indexed using array notation

myList = [1, 2, 3]
print("0th Value: %d" % myList[0]) 
myList.append(4)
print("List Lenght: %d" % len(myList)) 

for value in myList:
    print(value)


# Dictionaries
# Dictionaries are the mappings of the names to values, like key-value pairs/
#    Note that the use of the curlyt bracket and colon notations when defining the dictionary

myDict = {'a': 1, 'b': 2, 'c': 3}

print("A Value: %d" % myDict['a'])

myDict['a'] = 11

print("A Value: %d" % myDict['a'])

print("Keys: %s" % myDict.keys())

print("Values: %s" % myDict.values())

for key in myDict.keys():
    print(myDict[key])





















