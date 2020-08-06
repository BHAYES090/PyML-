#all returned values stand stand without print()
import numpy as np
from scipy import sparse

#Create a Vector

vector_row = np.array([1, 2, 3])

vector_column = np.array([[1],
                          [2],
                          [3]])

print(vector_row, "Vector Rows")
print(vector_column, "Vector Columns")

##NumPy’s main data structure is the multidimensional array.
##To create a vector, we simply create a one-dimensional array.
##Just like vectors, these arrays can be represented horizontally
##(i.e., rows) or vertically (i.e., columns).


##################################################################

#Create a Matrix

matrix0 = np.array([[1, 2],
                   [1, 2],
                   [1, 2]])

print(matrix0, "Matrix Creation")

##You may also use np.mat([i0], [i1]) 
##matrix data structure not recommended as arrays are the defacto standard data 
##structure of numpy and also numpy returns arrays and not matrices or matrix objects


##################################################################


#create a compressed sparse row
#scipy and sparse are needed to preform the following actions

matrix1 = np.array([[0, 0],
                   [0, 1],
                   [3, 0]])
#create a compressed sparse row (CSR) matrix

matrix_sparse = sparse.csr_matrix(matrix1)

#View sparse matrix
print(matrix_sparse, "Matrix sparsing on a small data set")

#Create larger matrix
matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Create compressed sparse row (CSR) matrix
matrix_large_sparse = sparse.csr_matrix(matrix_large)

print(matrix_large_sparse, "Matrix Sparsing on large data set")

##A sparse matrix or sparse array is a matrix in which most of the elements are zero


################################################################

#Select one or more elements from a Vector or Matrix

# Create row vector
vector = np.array([1, 2, 3, 4, 5, 6])

# Create matrix
matrix2 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Select third element of vector
print(vector[2], "Selecting element from a vector")

# Select second row, second column
print(matrix2[1,1], "Selecting element from a matrix")

# Select all elements of a vector
print(vector[:], "Select all elements of a vector")

# Select everything up to and including the third element
print(vector[:3], "Select everything up to and including the 3rd element")

# Select everything after the third element
print(vector[3:], "Select everything after the 3rd element")

# Select the last element
print(vector[-1], "Select the last element")

# Select the first two rows and all columns of a matrix
print(matrix2[:2,:], "Select the first two rows and all columns of a matrix")

# Select all rows and the second column
print(matrix2[:,1:2], "Select all rows and the second column")


################################################################


#View the shape, size and dimensions of a matrix

matrix3 = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# View number of rows and columns
print(matrix3.shape, "Shape")

# View number of elements (rows * columns)
print(matrix3.size, "Size")

# View number of dimensions
print(matrix3.ndim, "Number of Dimensions")


##############################################################

#Apply some function to multiple elements in an array or matrix

# Create matrix
matrix4 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Create function that adds 100 to something
add_100 = lambda i: i + 100

# Create vectorized function
vectorized_add_100 = np.vectorize(add_100)

# Apply function to all elements in matrix
vectorized_add_100(matrix4)

print(vectorized_add_100(matrix4), "Vectorized")

print(matrix4 + 100, "Broadcasting")

##NumPy’s vectorize class converts a function into a function that can apply to all
##elements in an array or slice of an array. It’s worth noting that vectorize is essentially a
##for loop over the elements and does not increase performance. Furthermore, NumPy arrays allow us
##to perform operations between arrays even if their dimensions are not the same
##(a process called broadcasting). For example, we can create a much simpler version
##of our solution using broadcasting:

##############################################################

#Find the maximum or minimum values

matrix5 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Return maximum element
print(np.max(matrix5), "Max value of the matrix")

# Return the minimum element
print(np.min(matrix5), "Select the minimum Value")

# find the maximum element in each column
print(np.max(matrix5, axis=0), "Use axis parameter to select max of each column ")

# find the maximum element in each row
print(np.max(matrix5, axis=1), "Use axis parameter to select max of each row")

# find the minimum element in each row
print(np.min(matrix5, axis=0), "Axis for min of each row")

# fing the minimum element in each column
print(np.min(matrix5, axis=1), "Axis for min of each column")


##############################################################


# Calculate the average, variance and standard deviation

# Create matrix
matrix6 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Return mean
print(np.mean(matrix6), "Mean")

# Return variance
print(np.var(matrix6), "Variance")

#Return standard deviation
print(np.std(matrix6), "Standard Deviation")

#Preform the operations along an axis
print(np.mean(matrix6, axis=0), "Preform mean calulation for each row")

#Preform the operations along an axis
print(np.mean(matrix6, axis=1), "Preform mean calulation for each column")


##############################################################

#change the shape (number of rows and columns) of an array
#without changing the element values

# Create 4x3 matrix
matrix7 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# Reshape matrix into 2x6 matrix
print(matrix7.reshape(2, 6), "reshaped matrix")

##reshape allows us to restructure an array so that we maintain the same
##data but it is organized as a different number of rows and columns.
##The only requirement is that the shape of the original and new matrix contain
##the same number of elements (i.e., the same size). We can see the size of a matrix using size:

print(matrix7.size, "View the size of a matrix")

##One useful argument in reshape is -1, which effectively means "as many as needed,"
##so reshape(-1, 1) means one row and as many columns as needed:

print(matrix7.reshape(-1, 1), "View the matrix reshaped to -1")

##if one element is provided, or one integer, reshape will return a 1-dimensional array of that length

print(matrix7.reshape(12), "Reshaped to one integer")


################################################################

#Transpose a Vector or Matrix

# Create matrix
matrix8 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Transpose matrix
print(matrix8.T, "Transposed matrix")

#Transposing swaps the column and row indices of each element. Technically a vector cannot be transposed
#because it is just a collection of values

#create a vector to transpose
print(np.array([1, 2, 3, 4, 5, 6]).T, "Attempting to transpose a vector")

#Transposing a vector is commonly referred to as transposing a row vector to a column vector
#this can be completed by adding a second pair of braces to the previous command

print(np.array([[1, 2, 3, 4, 5, 6]]).T, "Transpose a vector from a row to a column")

