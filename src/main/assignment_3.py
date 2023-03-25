import numpy as np

# Define the differential equation function
def func(t, y):
    return t - y**2
# Define the initial conditions and parameters
t0 = 0
y0 = 1
t_max = 2
n = 10
h = (t_max - t0) / n
# Implement Euler's method
t = t0
y = y0
for i in range(n):
    y += h * func(t, y)
    t += h
print(y)
print("\n")
# Implement Runge-Kutta method
t = t0
y = y0
for i in range(n):
    k1 = h * func(t, y)
    k2 = h * func(t + h/2, y + k1/2)
    k3 = h * func(t + h/2, y + k2/2)
    k4 = h * func(t + h, y + k3)
    y += (k1 + 2*k2 + 2*k3 + k4) / 6
    t += h
print(y)
print("\n")


# Define the augmented matrix
A = [[2, -1, 1, 6],
     [1, 3, 1, 0],
     [-1, 5, 4, -3]]
# Perform Gaussian elimination
for i in range(len(A)):
    # Find the row with the largest absolute value in the i-th column
    max_row = i
    for j in range(i+1, len(A)):
        if abs(A[j][i]) > abs(A[max_row][i]):
            max_row = j
    # Swap the current row with the row with the largest absolute value in the i-th column
    A[i], A[max_row] = A[max_row], A[i]
    # Reduce the i-th column to 1 by dividing the i-th row by A[i][i]
    pivot = A[i][i]
    for j in range(i, len(A[i])):
        A[i][j] /= pivot
    # Eliminate the i-th column in all other rows
    for j in range(len(A)):
        if j != i:
            factor = A[j][i]
            for k in range(i, len(A[i])):
                A[j][k] -= factor * A[i][k]
# Extract the solutions from the augmented matrix
x = [int(row[-1]) for row in A]
# Print the solutions
print(x)
print("\n")


def lu_decomposition(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            U[i][j] = matrix[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]
        for j in range(i + 1, n):
            L[j][i] = matrix[j][i]
            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]
            L[j][i] /= U[i][i]
    
    return L, U
def determinant(matrix):
    L, U = lu_decomposition(matrix)
    det = np.prod(np.diag(U))
    return det

A = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]])
# Calculate the determinant, L matrix, and U matrix
det = determinant(A)
L, U = lu_decomposition(A)
# Print the results
print(det)
print("\n")
print(L)
print("\n")
print(U)
print("\n")


def diagDM(m, n) :
 for i in range(0, n) :  
  sum = 0
  for j in range(0, n) :
   sum = sum + abs(m[i][j]) 
  sum = sum - abs(m[i][i])
  if (abs(m[i][i]) < sum) :
   return False
 return True
n = 5
m = [[ 9, 0, 5, 2, 1 ],
 [ 3, 9, 1, 2, 2 ],
 [ 0, 1, 7, 2, 3 ],
 [ 4, 2, 3, 12, 2],
 [ 3, 2, 4, 0, 8]]
if((diagDM(m, n))) :
 print ("True")
else :
 print ("False")
 
