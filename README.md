### Linear Algebra for Machine Learning and Data Science
    - Two lines of system of linear equations: intecepting point is the unique solution
      and the system is Complete and Non-singular.
    - when they line up on the same line, there are infinite solutions and the system is
      Redundant and singular.
    - when they are parallel, there is no solution and the system is Contradictory and singular.

### Linearly Dependent and Independent
    - Linearly Dependent: one vector can be represented by a linear combination of the other vectors.
    - Linearly Independent: no vector can be represented by a linear combination of the other vectors.
    - Linearly Dependent: the determinant of the matrix is zero.
    - Linearly Independent: the determinant of the matrix is non-zero.

### Determinant (ad - bc ) = 0
    - if the determinant is zero, the matrix is singular and the system is linearly dependent.
    - det /= 0  => Non-singular matrix has determint diffrent from zero.
    - det == 0  => Singular matrix has determinant equal to zero.
      
       a   b
       c   d      determinant = ad - bc

### NoteBook reference:
    - Open: File → Open
### Numpy Array
    - Import numpy as np  
    - Installation: python3 -m pip install numpy
    - import numpy as np

```python
- One dimensional array
one_dimensional_arr = np.array([10, 12])
   
```
```python
- Two dimensional array
two_dimensional_arr = np.array([[10, 12], [20, 30]])
```
```python
- Three dimensional array
three_dimensional_arr = np.array([[[10, 12], [20, 30]], [[40, 50], [60, 70]]])
```
```python
- Create an array that starts from the integer 1, ends at 20, incremented by 3.
np.arange(1, 20, 3)  
```
```python
- linspace: Create an evenly number starting with 0 end 100 and contain 5 elements.
- Create an evenly number starting with 0 end 100 and contain 5 elements.
np.linspace(0, 100, 5)
```
 
```python 
- dtype: Change the data type of the array. float64 is the default data type.
- By default the dtype of the created array is float64. but we can change it to int32.
np.linspace(0, 100, 5, dtype=np.int32)
```
    b_float = np.arange(3, dtype=float)
    print(b_float)

    output: [0. 1. 2.]

### Print the data type of the array
    print(b_float.dtype)
    output: float64

    char_arr = np.array(['Welcome to Math for ML!'])
    print(char_arr.dtype)
    print(char_arr.dtype) # Prints the data type of the array

    ['Welcome to Math for ML!']
    <U23
### More on NumPy arrays
    One of the advantages of using NumPy is that you can easily create arrays with built-in functions such as:
    np.ones() - Returns a new array setting values to one.
    np.zeros() - Returns a new array setting values to zero.
    np.empty() - Returns a new uninitialized array.
    np.random.rand() - Returns a new array with values chosen at random.

```python
# Return a new array of shape 3, filled with ones. 
ones_arr = np.ones(3)
print(ones_arr)
```

```python
# Return a new array of shape 3, filled with zeroes.
zeros_arr = np.zeros(3)
print(zeros_arr)
```
```python
# Return a new array of shape 3, without initializing entries.
empt_arr = np.empty(3)
print(empt_arr)
```
```python
# Return a new array of shape 3 with random numbers between 0 and 1.
rand_arr = np.random.rand(3)
print(rand_arr)
```
### Multidimensional Arrays
```python
# Create a 2 dimensional array (2-D)
two_dim_arr = np.array([[1,2,3], [4,5,6]])
print(two_dim_arr)
```
```python
# 1-D array 
one_dim_arr = np.array([1, 2, 3, 4, 5, 6])

# Multidimensional array using reshape()
multi_dim_arr = np.reshape(one_dim_arr, # the array to be reshaped
                          (2,3) # dimensions of the new array)
# Print the new 2-D array with two rows and three columns
print(multi_dim_arr)
```
### Finding size, shape and dimension
    ndarray.ndim - Stores the number dimensions of the array.
    ndarray.shape - Stores the shape of the array. Each number 
    in the tuple denotes the lengths of each corresponding dimension.
    ndarray.size - Stores the number of elements in the array.

```python
# Dimension of the 2-D array multi_dim_arr
multi_dim_arr.ndim
```
```python
# Shape of the 2-D array multi_dim_arr
# Returns shape of 2 rows and 3 columns
multi_dim_arr.shape
```
```python
# Size of the array multi_dim_arr
# Returns total number of elements
multi_dim_arr.size
```
### Array math operations
```python
arr_1 = np.array([2, 4, 6])
arr_2 = np.array([1, 3, 5])

# Adding two 1-D arrays
addition = arr_1 + arr_2
print(addition)

# Subtracting two 1-D arrays
subtraction = arr_1 - arr_2
print(subtraction)

# Multiplying two 1-D arrays elementwise
multiplication = arr_1 * arr_2
print(multiplication)
```
### Multiplying vector with a scalar (broadcasting)
```python
vector = np.array([1, 2])
vector * 1.6
```
### Indexing and slicing
    - Indexing: Accessing a single element of an array.
```python
# Select the third element of the array. Remember the counting starts from 0.
a = ([1, 2, 3, 4, 5])
print(a[2])

# Select the first element of the array.
print(a[0])
```
```python
# Indexing on a 2-D array
two_dim = np.array(([1, 2, 3],
          [4, 5, 6], 
          [7, 8, 9]))

# Select element number 8 from the 2-D array using indices i, j.
print(two_dim[2][1])
```
### Slicing: Accessing a subset of elements in an array.
    Slicing gives you a sublist of elements that you specify from the array.
    The slice notation specifies a start and end value, and copies the list 
    from start up to but not including the end (end-exclusive).
    
    The syntax is:
    array[start:end:step]
    
    If no value is passed to start, it is assumed start = 0, if no value is
    passed to end, it is assumed that end = length of array - 1 and if no value 
    is passed to step, it is assumed step = 1.
```python
# Slice the array a to get the array [2,3,4]
sliced_arr = a[1:4]
print(sliced_arr)
```
```python
# Slice the array a to get the array [1,2,3]
sliced_arr = a[:3]
print(sliced_arr)
```
```python
# Slice the array a to get the array [3,4,5]
sliced_arr = a[2:]
print(sliced_arr)
```
```python
# Slice the array a to get the array [1,3,5]
sliced_arr = a[::2]
print(sliced_arr)
```
```python
# Note that a == a[:] == a[::]
print(a == a[:] == a[::])
```
```python
# Slice the two_dim array to get the first two rows
sliced_arr_1 = two_dim[0:2]
sliced_arr_1
```
```python
# Similarily, slice the two_dim array to get the last two rows
sliced_two_dim_rows = two_dim[1:3]
print(sliced_two_dim_rows)
```
```python
sliced_two_dim_cols = two_dim[:,1]
print(sliced_two_dim_cols)
```
###  Stacking arrays
    - Stacking arrays horizontally
    Finally, stacking is a feature of NumPy that leads to increased customization of arrays. It means to join two or more arrays, either horizontally or vertically, meaning that it is done along a new axis.

    np.vstack() - stacks vertically
    np.hstack() - stacks horizontally
    np.hsplit() - splits an array into several smaller arrays
```python
a1 = np.array([[1,1], [2,2]])
a2 = np.array([[3,3], [4,4]])

print(f'a1:\n{a1}')
print(f'a2:\n{a2}')
```
```python
# Stack the arrays vertically
vert_stack = np.vstack((a1, a2))
print(vert_stack)
```
```python
#Stack the arrays horizontally
horz_stack = np.hstack((a1, a2))
print(horz_stack)
```
    ```
    Problem 1: You’re trying to figure out the price of apples, bananas, and cherries at the 
    store. You go three days in a row and bring this information:
    Day 1: You bought an apple, a banana, and a cherry, and paid $10.
    Day 2: You bought an apple, two bananas, and a cherry, and paid $15.
    Day 3: You bought an apple, a banana, and two cherries, and paid $12.
    Assume prices do not change between days.
    How much does each fruit cost?
    Apple=$3, Banana=$5, Cherry=$2
    Apple=$8, Banana=$2, Cherry=$3
    There must have been a mistake.
    Apple=$4, Banana=$5, Cherry=$1
    There is not enough information.
    ````
```python
from sympy import symbols, Eq, solve

# Define the symbols
apple, banana, cherry = symbols('apple banana cherry')

# Define the equations based on the given information
eq1 = Eq(apple + banana + cherry, 10)
eq2 = Eq(apple + 2*banana + cherry, 15)
eq3 = Eq(apple + banana + 2*cherry, 12)

# Solve the equations
solution = solve((eq1, eq2, eq3), (apple, banana, cherry))
solution
```

### Singular VS Non-singular Matrix

     - Unique solution: the system is Complete and Non-singular.  => Linearly Independent
     - Infinite solutions: the system is Redundant and singular.  => Linearly Dependent
     - No solution: the system is Contradictory and singular.     => Linearly Dependent
     - Singular matrix: the determinant of the matrix is zero.
     - Non-singular matrix: the determinant of the matrix is non-zero.

### Rows and column Linearly dependency and independence
    - Linearly Dependent: one vector can be represented by a linear combination of the other vectors.
    - Linearly Independent: no vector can be represented by a linear combination of the other vectors.
    - Linearly Dependent: the determinant of the matrix is zero.
    - Linearly Independent: the determinant of the matrix is non-zero.
    - If we can represent the third row or the third column of a matrix as the sum of the other rows or columns, 
      then the matrix is singular and the system is linearly dependent, Otherwise, the matrix is non-singular and
      the system is linearly independent.
---
### Solving Linear Systems: 2 variables
    - Representing and Solving System of Linear Equations using Matrices and Vectors
      Solving Systems of Linear Equations using Matrices
      Linear systems with two equations are easy to solve manually, but preparing for more complicated cases, 
      you will investigate some solution techniques.
    
      NumPy linear algebra package provides quick and reliable way to solve the system of linear equations using 
      function np.linalg.solve(A, b). Here  𝐴
      is a matrix, each row of which represents one equation in the system and each column corresponds to the variable  𝑥1
     ,  𝑥2
     . And  𝑏
      is a 1-D array of the free (right side) coefficients. More information about the np.linalg.solve() function can 
      be found in documentation.
    links: https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html

 Example:  −𝑥1+3𝑥2=7, 3𝑥1+2𝑥2=1 Lets us solve this system using NumPy.
```python
A = np.array([
        [-1, 3],
        [3, 2]
    ], dtype=np.dtype(float))

b = np.array([7, 1], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)
```
```python
  Check the dimensions of  𝐴 and  𝑏 using the shape attribute (you can also use np.shape() as an alternative):
print(f"Shape of A: {A.shape}")
print(f"Shape of b: {b.shape}")

# print(f"Shape of A: {np.shape(A)}")
# print(f"Shape of A: {np.shape(b)}")
```
```python
Now simply use np.linalg.solve(A, b) function to find the solution of the system  (1)
 . The result will be saved in the 1-D array  𝑥
 . The elements will correspond to the values of  𝑥1
  and  𝑥2
 :
x = np.linalg.solve(A, b)

print(f"Solution: {x}")
```
### Evaluating Determinant of a Matrix
    Matrix  𝐴
    corresponding to the linear system  (1)
    is a square matrix - it has the same number of rows and columns. In case of a square matrix it is possible to 
    calculate its determinant - a real number which characterizes some properties of the matrix. Linear system containing
    two (or more) equations with the same number of unknown variables will have one solution if and only if matrix  𝐴
    has non-zero determinant.

    Let's calculate the determinant using NumPy linear algebra package. You can do it with the np.linalg.det(A) 
    function. More information about it can be found in documentation.
    Documentation: https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html

### Solving System of Linear Equations using Elimination Method
    Elimination method is a technique of solving a system of linear equations. It is based on the idea of 
    performing elementary row operations to transform the matrix  𝐴
    into a matrix in row echelon form. The system of linear equations has the same solution as the system of 
    equations obtained by replacing  𝐴
    with its row echelon form. The system of equations in row echelon form is easy to solve because it is triangular.

    The elementary row operations are:
    - Interchange two rows.
    - Multiply a row by a non-zero number.
    - Add a multiple of a row to another row.

    The elimination method consists of two steps:
    - Forward elimination: transform the matrix  𝐴
    into a matrix in row echelon form.
    - Back substitution: solve the system of equations obtained by replacing  𝐴
    with its row echelon form.

    Let's solve the system of linear equations using elimination method. We will use the same system of equations 
    as in the previous example:  −𝑥1+3𝑥2=7, 3𝑥1+2𝑥2=1

    - consider this matrix: {−𝑥1+3𝑥2=7,3𝑥1+2𝑥2=1}
    - Preparation for the Implementation of Elimination Method in the Code
    - Representing the system in a matrix form as 
     you can apply the same operations to the rows of the matrix with Python code.
    - Unify matrix  𝐴 and array  𝑏 into one matrix using np.hstack() function. Note that the shape of the 
     originally defined array  𝑏 was  (2,), to stack it with the  (2,2) matrix you need to use .reshape((2, 1)) function:
```python
A_system = np.hstack((A, b.reshape((2, 1))))
print(A_system)
```
```python
    - Extract the first row of the matrix  𝐴
      print(A_system[1])
    - Extract the first column of the matrix  𝐴
```
```python
     print(A_system[:, 1])
    - Extract the first element of the matrix  𝐴
      print(A_system[0, 0])
    - Extract the second element of the matrix  𝐴
      print(A_system[0, 1])
    - Extract the third element of the matrix  𝐴
      print(A_system[0, 2])
```
```python
- Extract the first element of the array  𝑏
  print(A_system[0, 2])
- Extract the second element of the array  𝑏
  print(A_system[1, 2])
```
```python
- Extract the first row of the matrix  𝐴
  print(A_system[0])
- Extract the second row of the matrix  𝐴
  print(A_system[1])
```
```python
- Extract the first column of the matrix  𝐴
  print(A_system[:, 0])
- Extract the second column of the matrix  𝐴
  print(A_system[:, 1])
```
```python
- Extract the first two rows of the matrix  𝐴
  print(A_system[:2])
- Extract the first two columns of the matrix  𝐴
  print(A_system[:, :2])
```
```python
- Extract the first two rows and the first two columns of the matrix  𝐴
  print(A_system[:2, :2])
```
```python
- Extract the first two rows and the last column of the matrix  𝐴
  print(A_system[:2, 2])
```
```python
- Extract the last row of the matrix  𝐴
  print(A_system[-1])
- Extract the last column of the matrix  𝐴
  print(A_system[:, -1])
```
```python
- Extract the last two rows of the matrix  𝐴
  print(A_system[-2:])
- Extract the last two columns of the matrix  𝐴
  print(A_system[:, -2:])
```
```python
- Extract the last two rows and the last two columns of the matrix  𝐴
  print(A_system[-2:, -2:])
```
```python
- Extract the last two rows and the first column of the matrix  𝐴
  print (A_system[-2:, 0])
```
### Implementation of Elimination Method
```python
# Function .copy() is used to keep the original matrix without any changes.
A_system_res = A_system.copy()

A_system_res[1] = 3 * A_system_res[0] + A_system_res[1]

print(A_system_res)
```
```python
  - Multipy second row by  1/11
      A_system_res[1] = 1/11 * A_system_res[1]
      print(A_system_res)
```
### Graphical Representation of the Solution
    - Graphical Representation of the Solution
    - The solution of the system of linear equations is the point of intersection of the lines corresponding to the equations. 
      Let's plot the lines corresponding to the equations  −𝑥1+3𝑥2=7, 3𝑥1+2𝑥2=1
      The solution of the system of linear equations is the point of intersection of the lines corresponding to the equations. 
      Let's plot the lines corresponding to the equations  −𝑥1+3𝑥2=7, 3𝑥1+2𝑥2=1
```python
import matplotlib.pyplot as plt

def plot_lines(M):
    x_1 = np.linspace(-10,10,100)
    x_2_line_1 = (M[0,2] - M[0,0] * x_1) / M[0,1]
    x_2_line_2 = (M[1,2] - M[1,0] * x_1) / M[1,1]
    
    _, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_1, x_2_line_1, '-', linewidth=2, color='#0075ff',
        label=f'$x_2={-M[0,0]/M[0,1]:.2f}x_1 + {M[0,2]/M[0,1]:.2f}$')
    ax.plot(x_1, x_2_line_2, '-', linewidth=2, color='#ff7300',
        label=f'$x_2={-M[1,0]/M[1,1]:.2f}x_1 + {M[1,2]/M[1,1]:.2f}$')

    A = M[:, 0:-1]
    b = M[:, -1::].flatten()
    d = np.linalg.det(A)

    if d != 0:
        solution = np.linalg.solve(A,b) 
        ax.plot(solution[0], solution[1], '-o', mfc='none', 
            markersize=10, markeredgecolor='#ff0000', markeredgewidth=2)
        ax.text(solution[0]-0.25, solution[1]+0.75, f'$(${solution[0]:.0f}$,{solution[1]:.0f})$', fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-10, 10))
    ax.set_yticks(np.arange(-10, 10))

    plt.xlabel('$x_1$', size=14)
    plt.ylabel('$x_2$', size=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.axis([-10, 10, -10, 10])

    plt.grid()
    plt.gca().set_aspect("equal")

    plt.show()
plot_lines(A_system)
```
### System of Linear Equations with No Solutions
    - Given another matrix: {−𝑥1+3𝑥2=7,3𝑥1−9𝑥2=1}
    - let's find the determinant of the corresponding matrix.
```python
A_2 = np.array([[-1, 3],[3, -9]], dtype=np.dtype(float))
b_2 = np.array([7, 1], dtype=np.dtype(float))
d_2 = np.linalg.det(A_2)
print(f"Determinant of matrix A_2: {d_2:.2f}")

NB:It is equal to zero, thus the system cannot have one unique solution. It will have 
either infinitely many solutions or none. The consistency of it will depend on the 
free coefficients (right side coefficients). You can run the code in the following 
cell to check that the np.linalg.solve() function will give an error due to singularity.
```
```python
    - Finding singularity of a matrix
   try:
      x_2 = np.linalg.solve(A_2, b_2)
    except np.linalg.LinAlgError as err:
    print(err)
    
   - Prepare to apply the elimination method, constructing the matrix, corresponding to this linear system:

    A_2_system = np.hstack((A_2, b_2.reshape((2, 1))))
    print(A_2_system)

   - Perform elimination method:
    # copy() matrix.
    A_2_system_res = A_2_system.copy()
    
    # Multiply row 0 by 3 and add it to the row 1.
    A_2_system_res[1] = 3 * A_2_system_res[0] + A_2_system_res[1]
    print(A_2_system_res)
```
### System of Linear Equations with Infinite Number of Solutions
    - Changing free coefficients of the system  (5) you can bring it to consistency:
```python
   b_3 = np.array([7, -21], dtype=np.dtype(float))
   
-Prepare the new matrix, corresponding to the system  (6)
   A_3_system = np.hstack((A_2, b_3.reshape((2, 1))))
   print(A_3_system)
   
- Perform elimination method:
# copy() matrix.
A_3_system_res = A_3_system.copy()

# Multiply row 0 by 3 and add it to the row 1.
A_3_system_res[1] = 3 * A_3_system_res[0] + A_3_system_res[1]
print(A_3_system_res)
plot_lines(A_3_system)
```