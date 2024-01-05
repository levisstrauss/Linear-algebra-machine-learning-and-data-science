# Week 1: System of Linear equation

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

# Week 2: Solving system of linear equations

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
### Solving systems equation of three unknowns
    -1. We know how to solve a system of two equations with two unknowns.
    -2. To solve the equations with three unknowns, we need to add one more equation.
        - Divide each row of coefficients of a to eliminate
        - Use the first equation to remove a from the second and third equations
        - Solve the others 2 equations with 2 unknowns
        - Substitute the values of y and z into the first equation to find x

### Row Reductions
    - Row Reductions: The process of applying elementary row operations to a matrix to obtain a row echelon form.
    - Row Echelon Form: A matrix is in row echelon form if it satisfies the following conditions:
        - All rows consisting entirely of zeros are at the bottom of the matrix.
        - The first non-zero element in each row is a 1 (called a leading 1).
        - The leading 1 in each row is to the right of the leading 1 in the row above it.
        - All elements above and below leading 1s are zeros.
    - Reduced Row Echelon Form: A matrix is in reduced row echelon form if it satisfies the following conditions:
        - It is in row echelon form.
        - Each leading 1 is the only non-zero element in its column.
    - Pivot: A leading 1 in a matrix in row echelon form.
    - Pivot Column: A column that contains a pivot.
    - Pivot Position: The position of a pivot in a matrix.
    - Pivot Variable: A variable corresponding to a pivot column in the coefficient matrix of a system of equations.
    - Free Variable: A variable that is not a pivot variable.
    - Basic Variables: The variables corresponding to the pivot columns in the coefficient matrix of a system of equations.
    - Nonbasic Variables: The variables that are not basic variables.
    - Parametric Vector Form: A vector form of the solution to a system of equations that contains free variables.
    - Parametric Solution Set: The set of all solutions to a system of equations that contains free variables.
    - Homogeneous System: A system of equations in which all the constant terms are zero.
    - Trivial Solution: The solution to a homogeneous system of equations in which all the variables are equal to zero.
    - Nontrivial Solution: A solution to a homogeneous system of equations in which at least one variable is not equal to zero.
    - Inconsistent System: A system of equations that has no solution.
    - Consistent System: A system of equations that has at least one solution.
    - Elementary Matrix: A matrix that is obtained by performing an elementary row operation on an identity matrix.
    - Invertible Matrix: A square matrix that has an inverse.
    - Inverse Matrix: A matrix that, when multiplied by another matrix, results in the identity matrix.
    - Identity Matrix: A
### Row operations
    - Row operations: The process of applying elementary row operations to a matrix to obtain a row echelon form.
    - Row Echelon Form: A matrix is in row echelon form if it satisfies the following conditions:
        - All rows consisting entirely of zeros are at the bottom of the matrix.
        - The first non-zero element in each row is a 1 (called a leading 1).
        - The leading 1 in each row is to the right of the leading 1 in the row above it.
        - All elements above and below leading 1s are zeros.
    - Reduced Row Echelon Form: A matrix is in reduced row echelon form if it satisfies the following conditions:
        - It is in row echelon form.
        - Each leading 1 is the only non-zero element in its column.
    - Pivot: A leading 1 in a matrix in row echelon form.
    - Pivot Column: A column that contains a pivot.
    - Pivot Position: The position of a pivot in a matrix.
    - Pivot Variable: A variable corresponding to a pivot column in the coefficient matrix of a system of equations.
    - Free Variable: A variable that is not a pivot variable.
    - Basic Variables: The variables corresponding to the pivot columns in the coefficient matrix of a system of equations.
    - Nonbasic Variables: The variables that are not basic variables.
    - Parametric Vector Form: A vector form of the solution to a system of equations that contains free variables.
    - Parametric Solution Set: The set of all solutions to a system of equations that contains free variables.
    - Homogeneous System: A system of equations in which all the constant terms are zero.
    - Trivial Solution: The solution to a homogeneous system of equations in which all the variables are equal to zero.
    - Nontrivial Solution: A solution to a homogeneous system of equations in which at least one variable is not equal to zero.
    - Inconsistent System: A system of equations that has no solution.
    - Consistent System: A system of equations that has at least one solution.
    - Elementary Matrix: A matrix that is obtained by performing an elementary row operation on an identity matrix.
    - Invertible Matrix: A square matrix that has an inverse.
    - Inverse Matrix: A matrix that, when multiplied by another matrix, results in the identity matrix.
    - Identity Matrix: A square

       -- They are three types of row operations:
          - Interchange two rows.
          - Multiply a row by a non-zero number.
          - Add a multiple of a row to another row.
       These preserves singularity and non-singularity of a matrix.
```python
 { x+y=4 −6x+2y=16}


from sympy import symbols, Eq, solve

# Define the symbols for x and y
x, y = symbols('x y')

# Define the system of equations
eq1_elimination = Eq(x + y, 4)
eq2_elimination = Eq(-6*x + 2*y, 16)

# Solve the system of equations using the method of elimination
solution_elimination = solve((eq1_elimination, eq2_elimination), (x, y))
solution_elimination


 For three unknowns, we need to add one more equation.
    - Divide each row of coefficients of a to eliminate
    - Use the first equation to remove a from the second and third equations
    - Solve the others 2 equations with 2 unknowns
    - Substitute the values of y and z into the first equation to find x

# Define the new matrix for which the determinant is to be calculated
matrix_question_3 = Matrix([
    [-3, 2, -5],
    [8, 2, 6],
    [1, -1, 2]
])

# Calculate the determinant of the matrix
determinant_question_3 = matrix_question_3.det()
determinant_question_3
```
    Therefore, the operations that do not change the singularity (or non-singularity) of the matrix are:
    
    Adding a row to another one.
    Switching rows.
    Multiplying a row by a nonzero scalar.

### Representing and Solving a System of Linear Equations using Matrices
    - Lets us consider this matrix: {4𝑥1−3𝑥2+𝑥3=−10,2𝑥1+𝑥2+3𝑥3=0,−𝑥1+2𝑥2−5𝑥3=17}
    - Solving Systems of Linear Equations using Matrices
    - Prepare the matrix
```python
A = np.array([
        [4, -3, 1],
        [2, 1, 3],
        [-1, 2, -5]
    ], dtype=np.dtype(float))

b = np.array([-10, 0, 17], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)
```
```python
Check the dimensions of  𝐴 and  𝑏 using shape() function:
print(f"Shape of A: {np.shape(A)}")
print(f"Shape of b: {np.shape(b)}")
```
```python
    Now use np.linalg.solve(A, b) function to find the solution of the system.
    The result will be saved in the 1-D array  𝑥
    The elements will correspond to the values of  𝑥1 𝑥2 and  𝑥3
    
x = np.linalg.solve(A, b)
print(f"Solution: {x}")
```
### Evaluating the Determinant of a Matrix
      Matrix  𝐴
      corresponding to the linear system  (1)
      is a square matrix - it has the same number of rows and columns. In the case of 
      a square matrix it is possible to calculate its determinant - a real number that 
      characterizes some properties of the matrix. A linear system containing three 
      equations with three unknown variables will have one solution if and only if the matrix  𝐴
      has a non-zero determinant.
      Let's calculate the determinant using np.linalg.det(A) function:
```python
d = np.linalg.det(A)
print(f"Determinant of matrix A: {d:.2f}")
```
### Solving System of Linear Equations using Row Reduction
    - Preparation for Row Reduction
      Here you can practice the row reduction method for the linear system with three variables. 
      To apply it, first, unify matrix  𝐴
      and array  𝑏 into one matrix using np.hstack() function. Note that the shape of the originally defined array  𝑏
      was  (3,) to stack it with the  (3,3) matrix you need to transform it so that it has the same number of 
      dimensions. You can use .reshape((3, 1)) function:
```python
A_system = np.hstack((A, b.reshape((3, 1))))
print(A_system)
```
### Functions for Elementary Operations
```python
# exchange row_num of the matrix M with its multiple by row_num_multiple
# Note: for simplicity, you can drop check if  row_num_multiple has non-zero value, which makes the operation valid
def MultiplyRow(M, row_num, row_num_multiple):
    # .copy() function is required here to keep the original matrix without any changes
    M_new = M.copy()
    M_new[row_num] = M_new[row_num] * row_num_multiple
    return M_new

print("Original matrix:")
print(A_system)
print("\nMatrix after its third row is multiplied by 2:")
# remember that indexing in Python starts from 0, thus index 2 will correspond to the third row
print(MultiplyRow(A_system,2,2))
```
```python
# multiply row_num_1 by row_num_1_multiple and add it to the row_num_2, 
# exchanging row_num_2 of the matrix M in the result
def AddRows(M, row_num_1, row_num_2, row_num_1_multiple):
    M_new = M.copy()
    M_new[row_num_2] = row_num_1_multiple * M_new[row_num_1] + M_new[row_num_2]
    return M_new

print("Original matrix:")
print(A_system)
print("\nMatrix after exchange of the third row with the sum of itself and second row multiplied by 1/2:")
print(AddRows(A_system,1,2,1/2))
```
```python
# exchange row_num_1 and row_num_2 of the matrix M
def SwapRows(M, row_num_1, row_num_2):
    M_new = M.copy()
    M_new[[row_num_1, row_num_2]] = M_new[[row_num_2, row_num_1]]
    return M_new

print("Original matrix:")
print(A_system)
print("\nMatrix after exchange its first and third rows:")
print(SwapRows(A_system,0,2))
```
### Row Reduction and Solution of the Linear System
    Now you can use the defined operations to bring the matrix into row reduced form.
    To do this manually, it is convenient to have  1
    or  −1 value in the first element of the first row (the arithmetics of operations is 
    easier then). Performing calculations in Python, won't provide much of a benefit, 
    but it is better to do that for illustration purposes. So, let's swap the first and third rows:
```python
A_ref = SwapRows(A_system,0,2)
# Note: ref is an abbreviation of the row echelon form (row reduced form)
print(A_ref)
```
```python
Now you would need to make such elementary operations, that the first elements in the 
second and third row become equal to zero:
# multiply row 0 of the new matrix A_ref by 2 and add it to the row 1
A_ref = AddRows(A_ref,0,1,2)
print(A_ref)
```
```python
# multiply row 0 of the new matrix A_ref by 4 and add it to the row 2
A_ref = AddRows(A_ref,0,2,4)
print(A_ref)
```
```python
The next step will be to perform an operation by putting the second element 
in the third row equal to zero:
# multiply row 1 of the new matrix A_ref by -1 and add it to the row 2
A_ref = AddRows(A_ref,1,2,-1)
print(A_ref)
```
```python
It is easy now to find the value of  𝑥3
  from the third row, as it corresponds to the equation  −12𝑥3=24
 . Let's divide the row by -12:
 
# multiply row 2 of the new matrix A_ref by -1/12
A_ref = MultiplyRow(A_ref,2,-1/12)
print(A_ref)
```
```python
Now the second row of the matrix corresponds to the equation  5𝑥2−7𝑥3=34
  and the first row to the equation  −𝑥1+2𝑥2−5𝑥3=17. Referring to the elements 
  of the matrix, you can find the values of  𝑥2 and  𝑥1
x_3 = -2
x_2 = (A_ref[1,3] - A_ref[1,2] * x_3) / A_ref[1,1]
x_1 = (A_ref[0,3] - A_ref[0,2] * x_3 - A_ref[0,1] * x_2) / A_ref[0,0]

print(x_1, x_2, x_3)
```
### System of Linear Equations with No Solutions
    - {𝑥1+𝑥2+𝑥3=2, 𝑥2−3𝑥3=1, 2𝑥1+𝑥2+5𝑥3=0}
```python
   let's find the determinant of the corresponding matrix.
A_2= np.array([
        [1, 1, 1],
        [0, 1, -3],
        [2, 1, 5]
    ], dtype=np.dtype(float))
b_2 = np.array([2, 1, 0], dtype=np.dtype(float))
d_2 = np.linalg.det(A_2)
print(f"Determinant of matrix A_2: {d_2:.2f}")
```
```python
    It is equal to zero, thus the system cannot have one unique solution. 
    It will have either infinitely many solutions or none. The consistency of it will 
    depend on the free coefficients (right side coefficients). You can uncomment and 
    run the code in the following cell to check that the np.linalg.solve() function will 
    g ive an error due to singularity.

# x_2 = np.linalg.solve(A_2, b_2)
```
```python
    - You can check the system for consistency using ranks, but this is out of scope 
    here (you can review this topic following the link). For now you can perform elementary 
    operations to see that this particular system has no solutions:
```python
A_2_system = np.hstack((A_2, b_2.reshape((3, 1))))
print(A_2_system)
```
```python
# multiply row 0 by -2 and add it to the row 1
A_2_ref = AddRows(A_2_system,0,2,-2)
print(A_2_ref)
```
```python
# add row 1 of the new matrix A_2_ref to the row 2
A_2_ref = AddRows(A_2_ref,1,2,1)
print(A_2_ref)
```

### System of Linear Equations with Infinite Number of Solutions
    {𝑥1+𝑥2+𝑥3=2,𝑥2−3𝑥3=1,2𝑥1+𝑥2+5𝑥3=3.(3)
```python
  Define the new array of free coefficients:
  b_3 = np.array([2, 1, 3])
``` 
```python
Prepare the new matrix, corresponding to the system  (3)
A_3_system = np.hstack((A_2, b_3.reshape((3, 1))))
print(A_3_system)
```
```python
# multiply row 0 of the new matrix A_3_system by -2 and add it to the row 2
A_3_ref = AddRows(A_3_system,0,2,-2)
print(A_3_ref)
```
```python
# add row 1 of the new matrix A_3_ref to the row 2
A_3_ref = AddRows(A_3_ref,1,2,1)
print(A_3_ref)
```
---
### The rank of the matrix
    - The rank of the matrix is the number of linearly independent rows or columns in the matrix.
    - Pixel are stored in a matrix and the rank of the matrix are the ralated space needed to store
      matrix.
    - That is why, it is very crucial to learn the techniques of reducing the rank of the matrix.
      and one of the techniques is to use is called singular value decomposition or linear short.
    - Since the rank is reduced,it will only take a little space to store the matrix.
    - The number of information that a system can carry represents the number of rank
    - A system is non singular if having many informations and singular if having less information.

    - In a matrix, row one is an information but if row 2 can be represented by row 1, 
      then row 2 is not an information.

- NB: The rank of the matrix of the number of 1 after reducing the matrix in row echelon form.
 
       - If we have  1 - 1  => Rank 2  => Non Singular: the row echelon form has only ones and no zeros.
       - If we have  1 - 0  => Rank 1  => Singular
       - If we have  0 - 0  => Rank 0  => Singular
- 
### Row echelon form and row reduced echelon forms
    - Find the det of a given matrix and determine if the matrix is 
      singular or non singular, row echelon form and row reduced echelon form.
      Find the rank of the matrix.
```python
# Define the matrix
matrix_question_6 = Matrix([
    [7, 5, 3],
    [3, 2, 5],
    [1, 2, 1]
])

# Calculate the determinant of the matrix
determinant_question_6 = matrix_question_6.det()

# Check if the matrix is in Row-echelon form or Reduced row-echelon form
is_row_echelon = matrix_question_6 == matrix_question_6.echelon_form()
is_reduced_row_echelon = matrix_question_6 == matrix_question_6.rref()[0]

determinant_question_6, is_row_echelon, is_reduced_row_echelon

rank_question_6 = matrix_question_6.rank()
rank_question_6


Classification of matrixes bases on their 

# Define the matrices
matrix_a = Matrix([
    [0, 1, 1],
    [2, 4, 2],
    [1, 2, 1]
])
matrix_b = Matrix([
    [7.5, 5, 12.5],
    [3, 2, 5],
    [0, 0, 0]
])
matrix_c = matrix_question_6  # Previously defined matrix

# Calculate the rank of each matrix
rank_a = matrix_a.rank()
rank_b = matrix_b.rank()
# Rank of matrix C is already known from the previous calculation (rank_question_6)

rank_a, rank_b, rank_question_6
```
    The rank of a matrix indicates the amount of linearly 
    independent information it contains
---
### First Programming complete assignments
```python 
2x1 - x2 + x3 + x4 = 6,  
x1 + 2x2 - x3 - x4 = 3,  
-x1 + 2x2 + 2x3 + 2x4 =14,
x1 - x2 + 2x3 + x4 = 8, 
```
```python
 Construct matrix  𝐴 and vector  𝑏 corresponding to the system of linear equations  (1)
 A = np.array([     
        [2, -1, 1, 1],
        [1, 2, -1, -1],
        [-1, 2, 2, 2],
        [1, -1, 2, 1]    
    ], dtype=np.dtype(float)) 
b = np.array([6, 3, 14, 8], dtype=np.dtype(float))
```
```python
  - Find the determinant 𝑑 of matrix A and the solution vector 𝑥 for the system of linear equations  (1)
 ### START CODE HERE ###
# determinant of matrix A
d = np.linalg.det(A)

# solution of the system of linear equations 
# with the corresponding coefficients matrix A and free coefficients b
x = np.linalg.solve(A, b)
### END CODE HERE ###

print(f"Determinant of matrix A: {d:.2f}")

print(f"Solution vector: {x}")
```
    Multiply any row by non-zero number
    Add two rows and exchange one of the original rows with the result of the addition
    Swap rows
```python
def MultiplyRow(M, row_num, row_num_multiple):
    # .copy() function is required here to keep the original matrix without any changes
    M_new = M.copy()     
    # exchange row_num of the matrix M_new with its multiple by row_num_multiple
    # Note: for simplicity, you can drop check if  row_num_multiple has non-zero value, which makes the operation valid
    M_new[row_num] = M_new[row_num] = M_new[row_num] * row_num_multiple
    return M_new
    
def AddRows(M, row_num_1, row_num_2, row_num_1_multiple):
    M_new = M.copy()     
    # multiply row_num_1 by row_num_1_multiple and add it to the row_num_2, 
    # exchanging row_num_2 of the matrix M_new with the result
    M_new[row_num_2] = row_num_1_multiple * M_new[row_num_1] + M_new[row_num_2]
    return M_new

def SwapRows(M, row_num_1, row_num_2):
    M_new = M.copy()     
    # exchange row_num_1 and row_num_2 of the matrix M_new
    M_new[[row_num_1, row_num_2]] = M_new[[row_num_2, row_num_1]]
    return M_new
```
```python
  - Check your result
A_test = np.array([
        [1, -2, 3, -4],
        [-5, 6, -7, 8],
        [-4, 3, -2, 1], 
        [8, -7, 6, -5]
    ], dtype=np.dtype(float))
print("Original matrix:")
print(A_test)

print("\nOriginal matrix after its third row is multiplied by -2:")
print(MultiplyRow(A_test,2,-2))

print("\nOriginal matrix after exchange of the third row with the sum of itself and first row multiplied by 4:")
print(AddRows(A_test,0,2,4))

print("\nOriginal matrix after exchange of its first and third rows:")
print(SwapRows(A_test,0,2))
```
    Apply elementary operations to the defined above matrix A, performing row reduction according to the given instructions.
    to swap row 1 and row 2 of matrix A, use the code SwapRows(A,1,2)
    to multiply row 1 of matrix A by 4 and add it to the row 2, use the code AddRows(A,1,2,4)
    to multiply row 2 of matrix A by 5, use the code MultiplyRow(A,2,5)
```python
  def augmented_to_ref(A, b):    
    ### START CODE HERE ###
    # stack horizontally matrix A and vector b, which needs to be reshaped as a vector (4, 1)
    A_system = np.hstack((A.astype(float), b.reshape(-1, 1)))
    
    # swap row 0 and row 1 of matrix A_system (remember that indexing in NumPy array starts from 0)
    A_system = SwapRows(A_system, 0, 1)
    
    # multiply row 0 of the new matrix A_ref by -2 and add it to the row 1
    A_system = AddRows(A_system, 0, 1, -2)
    
    # add row 0 of the new matrix A_ref to the row 2, replacing row 2
    A_system = AddRows(A_system, 0, 2, 1)
    
    # multiply row 0 of the new matrix A_ref by -1 and add it to the row 3
    A_system = AddRows(A_system, 0, 3, -1)
    
    # add row 2 of the new matrix A_ref to the row 3, replacing row 3
    A_system = AddRows(A_system, 2, 3, 1)
    
    # swap row 1 and 3 of the new matrix A_ref
    A_system = SwapRows(A_system, 1, 3)
    
    # add row 2 of the new matrix A_ref to the row 3, replacing row 3
    A_system = AddRows(A_system, 2, 3, 1)
    
    # multiply row 1 of the new matrix A_ref by -4 and add it to the row 2
    A_system = AddRows(A_system, 1, 2, -4)

    # add row 1 of the new matrix A_ref to the row 3, replacing row 3
    A_system = AddRows(A_system, 1, 3, 1)
    
    # multiply row 3 of the new matrix A_ref by 2 and add it to the row 2
    A_system = AddRows(A_system, 3, 2, 2)
    
    # multiply row 2 of the new matrix A_ref by -8 and add it to the row 3
    A_system = AddRows(A_system, 2, 3, -8)
    
    # multiply row 3 of the new matrix A_ref by -1/17
    A_system = MultiplyRow(A_system, 3, -1/17)
    ### END CODE HERE ###
    
    return A_system.astype(int)

A_ref = augmented_to_ref(A, b)

print(A_ref)
```
```python
  - Solution for the System of Equations using Row Reduction
  - Hint:  x1 + 2x2 - x3 - x4 = 3, 
           x2 + 4x3 + 3x4 = 22,
           x3 + 3x4 =7,
           x4 = 1, 

# find the value of x_4 from the last line of the reduced matrix A_ref
x_4 = 1

# find the value of x_3 from the previous row of the matrix. Use value of x_4.
x_3 = 7 - 3 * x_4

# find the value of x_2 from the second row of the matrix. Use values of x_3 and x_4
x_2 =  22 - 4 * x_3 - 3 * x_4

# find the value of x_1 from the first row of the matrix. Use values of x_2, x_3 and x_4
x_1 = 3 - (2 * x_2) + x_3 + x_4

print(x_1, x_2, x_3, x_4)
```
```python
 Using the same elementary operations as above you can reduce the matrix further to 
 diagonal form, from which you can see the solutions easily.

def ref_to_diagonal(A_ref):    
    ### START CODE HERE ###
    # multiply row 3 of the matrix A_ref by -3 and add it to the row 2
    A_diag = AddRows(A_ref, 3, 2, -3)
    
    # multiply row 3 of the new matrix A_diag by -3 and add it to the row 1
    A_diag = AddRows(A_diag, 3, 1, -3)
    
    # add row 3 of the new matrix A_diag to the row 0, replacing row 0
    A_diag = AddRows(A_diag, 3, 0, 1)
    
    # multiply row 2 of the new matrix A_diag by -4 and add it to the row 1
    A_diag = AddRows(A_diag, 2, 1, -4)
    
    # add row 2 of the new matrix A_diag to the row 0, replacing row 0
    A_diag = AddRows(A_diag, 2, 0, 1)
    
    # multiply row 1 of the new matrix A_diag by -2 and add it to the row 0
    A_diag = AddRows(A_diag, 1, 0, -2)
    ### END CODE HERE ###
    
    return A_diag
    
A_diag = ref_to_diagonal(A_ref)

print(A_diag)
```
# Week 3: Vectors and linear Transformations
  
### Vectors Operations
    - Addition of vectors
    Ex: 𝑎=(1,2,3)  and  𝑏=(4,5,6)  are two vectors of the same dimension.

    - Addition of a and b is defined as: 𝑐=𝑎+𝑏=(1+4,2+5,3+6)=(5,7,9)

    - Sousraction of vectors: 𝑐=𝑎−𝑏=(1−4,2−5,3−6)=(−3,−3,−3)

    - Multiplication of a and b is defined as: 𝑐=𝑎∗𝑏=(1∗4,2∗5,3∗6)=(4,10,18)

    - Division of a and b is defined as: 𝑐=𝑎/𝑏=(1/4,2/5,3/6)=(0.25,0.4,0.5)

    - Dot product of a and b is defined as: 𝑐=𝑎⋅𝑏=(1∗4)+(2∗5)+(3∗6)=(4+10+18)=32

    - Cross product of a and b is defined as: 𝑐=𝑎×𝑏=(2∗6−3∗5,3∗4−1∗6,1∗5−2∗4)=(12−15,12−6,5−8)=(−3,6,−3)

    - The dot product of two vectors is a scalar, while the cross product of two vectors is a vector.

    - The norm of a vector is defined as: 𝑎=√(𝑎1^2+b2^2)

    - The distance between vectors a and b: 𝑐=√((𝑎1−𝑏1)^2+(𝑎2−𝑏2)^2)

    - Two vectors are orthogonal if their dot product is zero.
    - The dot product of A is <A, A> = |A|^2
    - The dot product of A and B is <A, B> = 0 of A and B are orthogonal.

    u   /|
       / |   V
    - /__|________>   <U, V> = |U| |V| cos(theta)
       U'

### Vector Operations: Scalar Multiplication, Sum and Dot Product of Vectors

    - Visualization of a Vector  𝑣∈ℝ2
```python
import matplotlib.pyplot as plt

def plot_vectors(list_v, list_label, list_color):
    _, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-10, 10))
    ax.set_yticks(np.arange(-10, 10))
    
    
    plt.axis([-10, 10, -10, 10])
    for i, v in enumerate(list_v):
        sgn = 0.4 * np.array([[1] if i==0 else [i] for i in np.sign(v)])
        plt.quiver(v[0], v[1], color=list_color[i], angles='xy', scale_units='xy', scale=1)
        ax.text(v[0]-0.2+sgn[0], v[1]-0.2+sgn[1], list_label[i], fontsize=14, color=list_color[i])

    plt.grid()
    plt.gca().set_aspect("equal")
    plt.show()

v = np.array([[1],[3]])
# Arguments: list of vectors as NumPy arrays, labels, colors.
plot_vectors([v], [f"$v$"], ["black"])
```
    
```python
- Scalar Multiplication of a Vector
v = np.array([[1],[3]])
w = np.array([[4],[-1]])

plot_vectors([v, w, v + w], [f"$v$", f"$w$", f"$v + w$"], ["black", "black", "red"])
# plot_vectors([v, w, np.add(v, w)], [f"$v$", f"$w$", f"$v + w$"], ["black", "black", "red"])
``` 
```python
   The dot product
x = [1, -2, -5]
y = [4, 3, -1]

// Python version code
def dot(x, y):
    s=0
    for xi, yi in zip(x, y):
        s += xi * yi
    return s

print("The dot product of x and y is", dot(x, y))

// Numpy version code
print("np.dot(x,y) function returns dot product of x and y:", np.dot(x, y))

// Another way using numpy array

print("This line output is a dot product of x and y: ", np.array(x) @ np.array(y))

print("\nThis line output is an error:")
try:
    print(x @ y)
except TypeError as err:
    print(err) 
```
    - Dot Product using np.array  are vectorized operations so much powerful than for loops.
```python
import time

tic = time.time()
c = dot(a,b)
toc = time.time()
print("Dot product: ", c)
print ("Time for the loop version:" + str(1000*(toc-tic)) + " ms")

-------------------------
import time

tic = time.time()
c = dot(a,b)
toc = time.time()
print("Dot product: ", c)
print ("Time for the loop version:" + str(1000*(toc-tic)) + " ms")

--------------------------
tic = time.time()
c = a @ b
toc = time.time()
print("Dot product: ", c)
print ("Time for the vectorized version, @ function: " + str(1000*(toc-tic)) + " ms")
```
---
### Linear Transformation
    - A linear transformation is a function from one vector space to another that respects the   
      underlying (linear) structure of each vector space.
    - If A is a matrix of the form (ab, cd) the inverse of A is 

      A^-1 = 1/det(A) (d, -b, -c, a)

    - Non singular matrix always have an inverse det is different than 0
    - Singular matrix does not have an inverse. det is equal to 0

The Norm of a vector can be computed as
```python
# Define the vector v
v = np.array([1, -5, 2, 0, -3])

# Calculate the norm of the vector v
norm_v = np.linalg.norm(v)
norm_v
```
### Matrix Multiplication using Python
```python
A = np.array([[4, 9, 9], [9, 1, 6], [9, 2, 3]])
print("Matrix A (3 by 3):\n", A)

B = np.array([[2, 2], [5, 7], [4, 4]])
print("Matrix B (3 by 2):\n", B)

np.matmul(A, B) or A @ B
```
### Matrix Convention and Broadcasting
```python
try:
    np.matmul(B, A)
except ValueError as err:
    print(err)

try:
    B @ A
except ValueError as err:
    print(err)

x = np.array([1, -2, -5])
y = np.array([4, 3, -1])

print("Shape of vector x:", x.shape)
print("Number of dimensions of vector x:", x.ndim)
print("Shape of vector x, reshaped to a matrix:", x.reshape((3, 1)).shape)
print("Number of dimensions of vector x, reshaped to a matrix:", x.reshape((3, 1)).ndim)

np.matmul(x,y)
```
      You can see that there is no error and that the result is actually a dot product  𝑥⋅𝑦
      ! So, vector  𝑥
      was automatically transposed into the vector  1×3
      and matrix multiplication  𝑥𝑇𝑦
      was calculated. While this is very convenient, you need to keep in mind such 
      functionality in Python and pay attention to not use it in a wrong way. The following 
      cell will return an error:
```python
try:
    np.matmul(x.reshape((3, 1)), y.reshape((3, 1)))
except ValueError as err:
    print(err)
np.dot(A, B)
```

### Linear Transformation
    - A linear transformation is a function from one vector space to another that respects the   
    underlying (linear) structure of each vector space.
    A transformation is a function from one vector space to another that respects the underlying (linear) structure of each vector space. Referring to a specific transformation, you can use a symbol, such as  𝑇
    Specifying the spaces containing the input and output vectors, e.g.  ℝ2 and  ℝ3
    you can write  𝑇:ℝ2→ℝ3
    Transforming vector  𝑣∈ℝ2 into the vector  𝑤∈ℝ3 by the transformation  𝑇
    you can use the notation  𝑇(𝑣)=𝑤
    and read it as "T of v equals to w" or "vector w is an image of vector v with the transformation T".

    The following Python function corresponds to the transformation  𝑇:ℝ2→ℝ3
    with the following symbolic formula:

    T([v1, v2]) =     3v1

                   |   0     |
                   |         |
                   |_  -2v2 _|
```python
import numpy as np
# OpenCV library for image transformations.
import cv2

def T(v):
    w = np.zeros((3,1))
    w[0,0] = 3*v[0,0]
    w[2,0] = -2*v[1,0]
    
    return w

v = np.array([[3], [5]])
w = T(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)
```
### Linear Transformations
    A transformation 𝑇 is said to be linear if the following two properties are 
    true for any scalar 𝑘 and any input vectors 𝑢 and 𝑣:

    𝑇(𝑘𝑣)=𝑘𝑇(𝑣),
    𝑇(𝑢+𝑣)=𝑇(𝑢)+𝑇(𝑣)
```python
u = np.array([[1], [-2]])
v = np.array([[2], [4]])

k = 7

print("T(k*v):\n", T(k*v), "\n k*T(v):\n", k*T(v), "\n\n")
print("T(u+v):\n", T(u+v), "\n T(u)+T(v):\n", T(u)+T(v))
```
### Transformations Defined as a Matrix Multiplication

```python
def L(v):
    A = np.array([[3,0], [0,0], [0,-2]])
    print("Transformation matrix:\n", A, "\n")
    w = A @ v
    
    return w

v = np.array([[3], [5]])
w = L(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)   
```
### Standard Transformations in a Plane
```python
def T_hscaling(v):
    A = np.array([[2,0], [0,1]])
    w = A @ v
    
    return w
    
    
def transform_vectors(T, v1, v2):
    V = np.hstack((v1, v2))
    W = T(V)
    
    return W
    
e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

transformation_result_hscaling = transform_vectors(T_hscaling, e1, e2)

print("Original vectors:\n e1= \n", e1, "\n e2=\n", e2, 
      "\n\n Result of the transformation (matrix form):\n", transformation_result_hscaling)
```
```python
You can get a visual understanding of the transformation, producing a plot which 
displays input vectors, and their transformations. Do not worry if the code in the 
following cell will not be clear - at this stage this is not important code to understand.

import matplotlib.pyplot as plt

def plot_transformation(T, e1, e2):
    color_original = "#129cab"
    color_transformed = "#cc8933"
    
    _, ax = plt.subplots(figsize=(7, 7))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-5, 5))
    ax.set_yticks(np.arange(-5, 5))
    
    plt.axis([-5, 5, -5, 5])
    plt.quiver([0, 0],[0, 0], [e1[0], e2[0]], [e1[1], e2[1]], color=color_original, angles='xy', scale_units='xy', scale=1)
    plt.plot([0, e2[0], e1[0], e1[0]], 
             [0, e2[1], e2[1], e1[1]], 
             color=color_original)
    e1_sgn = 0.4 * np.array([[1] if i==0 else [i] for i in np.sign(e1)])
    ax.text(e1[0]-0.2+e1_sgn[0], e1[1]-0.2+e1_sgn[1], f'$e_1$', fontsize=14, color=color_original)
    e2_sgn = 0.4 * np.array([[1] if i==0 else [i] for i in np.sign(e2)])
    ax.text(e2[0]-0.2+e2_sgn[0], e2[1]-0.2+e2_sgn[1], f'$e_2$', fontsize=14, color=color_original)
    
    e1_transformed = T(e1)
    e2_transformed = T(e2)
    
    plt.quiver([0, 0],[0, 0], [e1_transformed[0], e2_transformed[0]], [e1_transformed[1], e2_transformed[1]], 
               color=color_transformed, angles='xy', scale_units='xy', scale=1)
    plt.plot([0,e2_transformed[0], e1_transformed[0]+e2_transformed[0], e1_transformed[0]], 
             [0,e2_transformed[1], e1_transformed[1]+e2_transformed[1], e1_transformed[1]], 
             color=color_transformed)
    e1_transformed_sgn = 0.4 * np.array([[1] if i==0 else [i] for i in np.sign(e1_transformed)])
    ax.text(e1_transformed[0]-0.2+e1_transformed_sgn[0], e1_transformed[1]-e1_transformed_sgn[1], 
            f'$T(e_1)$', fontsize=14, color=color_transformed)
    e2_transformed_sgn = 0.4 * np.array([[1] if i==0 else [i] for i in np.sign(e2_transformed)])
    ax.text(e2_transformed[0]-0.2+e2_transformed_sgn[0], e2_transformed[1]-e2_transformed_sgn[1], 
            f'$T(e_2)$', fontsize=14, color=color_transformed)
    
    plt.gca().set_aspect("equal")
    plt.show()
    
plot_transformation(T_hscaling, e1, e2)
```

###  Example 2: Reflection about y-axis (the vertical axis)
```python
Function T_reflection_yaxis() defined below corresponds to the reflection about y-axis:
def T_reflection_yaxis(v):
    A = np.array([[-1,0], [0,1]])
    w = A @ v
    
    return w
    
e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

transformation_result_reflection_yaxis = transform_vectors(T_reflection_yaxis, e1, e2)

print("Original vectors:\n e1= \n", e1,"\n e2=\n", e2, 
      "\n\n Result of the transformation (matrix form):\n", transformation_result_reflection_yaxis)
      
plot_transformation(T_reflection_yaxis, e1, e2)
```

### Application of Linear Transformations: Computer Graphics
```python
img = cv2.imread('images/leaf_original.png', 0)
plt.imshow(img)


image_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

plt.imshow(image_rotated)

Applying the shear you will get the following output:

rows,cols = image_rotated.shape
# 3 by 3 matrix as it is required for the OpenCV library, don't worry about the details of it for now.
M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
image_rotated_sheared = cv2.warpPerspective(image_rotated, M, (int(cols), int(rows)))
plt.imshow(image_rotated_sheared)

What if you will apply those two transformations in the opposite order? Do you think the result will be the same? 
Run the following code to check that:
image_sheared = cv2.warpPerspective(img, M, (int(cols), int(rows)))
image_sheared_rotated = cv2.rotate(image_sheared, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(image_sheared_rotated)
```
```python
M_rotation_90_clockwise = np.array([[0, 1], [-1, 0]])
M_shear_x = np.array([[1, 0.5], [0, 1]])

print("90 degrees clockwise rotation matrix:\n", M_rotation_90_clockwise)
print("Matrix for the shear along x-axis:\n", M_shear_x)

Now check that the results of their multiplications M_rotation_90_clockwise @ M_shear_x and 
M_shear_x @ M_rotation_90_clockwise are different:
print("M_rotation_90_clockwise by M_shear_x:\n", M_rotation_90_clockwise @ M_shear_x)
print("M_shear_x by M_rotation_90_clockwise:\n", M_shear_x @ M_rotation_90_clockwise)
```
---
### # Single Perceptron Neural Networks for Linear Regression
    Welcome to your week 3 programming assignment. Now you are ready to apply matrix 
    multiplication by building your first neural network with a single perceptron.
    After this assignment you will be able to:
    Implement a neural network with a single perceptron and one input node for simple 
    linear regression
    Implement forward propagation using matrix multiplication
    Implement a neural network with a single perceptron and two input nodes for multiple 
    linear regression
    Note: Backward propagation with the parameters update requires understanding of Calculus.
    It is discussed in details in the Course "Calculus" (Course 2 in the Specialization 
    
    "Mathematics for Machine Learning"). In this assignment backward propagation and 
    parameters update functions are hidden.


    - Let's first import all the packages that you will need during this assignment.
```python
import numpy as np
import matplotlib.pyplot as plt
# A function to create a dataset.
from sklearn.datasets import make_regression
# A library for data manipulation and analysis.
import pandas as pd
# Some functions defined specifically for this notebook.
import w3_tools

# Output of plotting commands is displayed inline within the Jupyter notebook.
%matplotlib inline 

# Set a seed so that the results are consistent.
np.random.seed(3) 
```
### Dataset
    First, let's get the dataset you will work on. The following code will create  𝑚=30
    data points  (𝑥1,𝑦1), ...,  (𝑥𝑚,𝑦𝑚) and save them in NumPy arrays X and Y of a shape  (1×𝑚)
 
```python
m = 30

X, Y = make_regression(n_samples=m, n_features=1, noise=20, random_state=1)

X = X.reshape((1, m))
Y = Y.reshape((1, m))

print('Training dataset X:')
print(X)
print('Training dataset Y')
print(Y)

-- Plot te dataset
plt.scatter(X,  Y, c="black")

plt.xlabel("$x$")
plt.ylabel("$y$")
```
    Exercise 1
    What is the shape of the variables X and Y? In addition, how many training 
    examples do you have?
```python
### START CODE HERE ### (~ 3 lines of code)
# Shape of variable X.
shape_X = X.shape
# Shape of variable Y.
shape_Y = Y.shape
# Training set size.
m = shape_X[1] # or shape_X[1] same
### END CODE HERE ###

print ('The shape of X: ' + str(shape_X))
print ('The shape of Y: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))
```
### Implementation of the Neural Network Model for Linear Regression
    Let's setup the neural network in a way which will allow to extend this simple case 
    of a model to more complicated structures later.

    Exercise 2
    Define two variables:
    
    n_x: the size of the input layer
    n_y: the size of the output layer
```python
# GRADED FUNCTION: layer_sizes

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """
    ### START CODE HERE ### (~ 2 lines of code)
    # Size of input layer.
    n_x = X.shape[0]
    # Size of output layer.
    n_y = Y.shape[0]
    ### END CODE HERE ###
    return (n_x, n_y)

(n_x, n_y) = layer_sizes(X, Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the output layer is: n_y = " + str(n_y))
```
    Exercise 3
    Initialize the model's parameters.
    Implement the function initialize_parameters().

    Instructions:
    Make sure your parameters' sizes are right. Refer to the neural network figure 
    above if needed.
    You will initialize the weights matrices with random values.
    Use: np.random.randn(a,b) * 0.01 to randomly initialize a matrix of shape (a,b).
    You will initialize the bias vectors as zeros.
    Use: np.zeros((a,b)) to initialize a matrix of shape (a,b) with zeros.
```python
# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_y):
    """
    Returns:
    params -- python dictionary containing your parameters:
                    W -- weight matrix of shape (n_y, n_x)
                    b -- bias value set as a vector of shape (n_y, 1)
    """
    
    ### START CODE HERE ### (~ 2 lines of code)
    W = np.random.randn(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))
    ### END CODE HERE ###
    
    assert (W.shape == (n_y, n_x))
    assert (b.shape == (n_y, 1))
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

parameters = initialize_parameters(n_x, n_y)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))
```
    Exercise 4
    Implement the forward propagation module.
    Complete the function forward_propagation().

    Instructions:
    Check the mathematical representation of the linear regression.
    You can use np.dot(A,B) to calculate the dot product of two matrices.
    You can use np.add(A,B) to calculate the sum of two matrices.
```python

def forward_propagation(X, parameters, n_y):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    Y_hat -- The output of size (n_y, m)
    """
    # Retrieve each parameter from the dictionary "parameters".
    ### START CODE HERE ### (~ 2 lines of code)
    W = parameters['W']
    b = parameters["b"]
    ### END CODE HERE ###
    
    # Implement Forward Propagation to calculate Z.
    ### START CODE HERE ### (~ 2 lines of code)
    Z = np.dot(W, X) + b
    Y_hat = Z
    ### END CODE HERE ###
    assert(Y_hat.shape == (n_y, X.shape[1]))
    return Y_hat
Y_hat = forward_propagation(X, parameters, n_y)

print(Y_hat)
```
    Remember that your weights were just initialized with some random values, so the model has not been trained yet.
    Define a cost function  (5) which will be used to train the model:
```python
def compute_cost(Y_hat, Y):
    """
    Computes the cost function as a sum of squares
    
    Arguments:
    Y_hat -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- sum of squares scaled by 1/(2*number of examples)
    
    """
    # Number of examples.
    m = Y.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    
    return cost

print("cost = " + str(compute_cost(Y_hat, Y)))


parameters = w3_tools.train_nn(parameters, Y_hat, X, Y)

print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))
```
###  Integrate parts 2.1, 2.2 and 2.3 in nn_model()
     Exercise 5 
     Build your neural network model in nn_model().
     Instructions: The neural network model has to use the previous 
     functions in the right order.
```python
# GRADED FUNCTION: nn_model

def nn_model(X, Y, num_iterations=10, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    num_iterations -- number of iterations in the loop
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to make predictions.
    """
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    
    # Initialize parameters
    ### START CODE HERE ### (~ 1 line of code)
    parameters = initialize_parameters(n_x, n_y)
    ### END CODE HERE ###
    
    # Loop
    for i in range(0, num_iterations):
         
        ### START CODE HERE ### (~ 2 lines of code)
        # Forward propagation. Inputs: "X, parameters, n_y". Outputs: "Y_hat".
        Y_hat = forward_propagation(X, parameters, n_y)
        
        # Cost function. Inputs: "Y_hat, Y". Outputs: "cost".
        cost = compute_cost(Y_hat, Y)
        ### END CODE HERE ###
        
        
        # Parameters update.
        parameters = w3_tools.train_nn(parameters, Y_hat, X, Y) 
        
        # Print the cost every iteration.
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

parameters = nn_model(X, Y, num_iterations=15, print_cost=True)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))

W_simple = parameters["W"]
b_simple = parameters["b"]
```
    You can see that after a few iterations the cost function does not change 
    anymore (the model converges).
    Note: This is a very simple model. In reality the models do not converge that quickly.
    The final model parameters can be used for making predictions. Let's plot the linear regression line and some predictions. The regression line is red and the predicted points are blue.
```python
X_pred = np.array([-0.95, 0.2, 1.5])

fig, ax = plt.subplots()
plt.scatter(X, Y, color = "black")

plt.xlabel("$x$")
plt.ylabel("$y$")
    
X_line = np.arange(np.min(X[0,:]),np.max(X[0,:])*1.1, 0.1)
ax.plot(X_line, W_simple[0,0] * X_line + b_simple[0,0], "r")
ax.plot(X_pred, W_simple[0,0] * X_pred + b_simple[0,0], "bo")
plt.plot()
plt.show()
```
### Multiple Linear Regression
    Models are not always as simple as the one above. In some 
    cases your output is dependent on more than just one variable.
    Let's look at the case where the output depends on two input variables.

### Dataset
```python
- Load the dataset
df = pd.read_csv('data/house_prices_train.csv')

X_multi = df[['GrLivArea', 'OverallQual']]
Y_multi = df['SalePrice']

- Look the dataset
print(f"X_multi:\n{X_multi}\n")
print(f"Y_multi:\n{Y_multi}\n")

- Normaliiize the dataset
X_multi_norm = (X_multi - np.mean(X_multi))/np.std(X_multi)
Y_multi_norm = (Y_multi - np.mean(Y_multi))/np.std(Y_multi)

- Convert results to the NumPy arrays, transpose

X_multi_norm = np.array(X_multi_norm).T
Y_multi_norm = np.array(Y_multi_norm).reshape((1, len(Y_multi_norm)))

print ('The shape of X: ' + str(X_multi_norm.shape))
print ('The shape of Y: ' + str(Y_multi_norm.shape))
print ('I have m = %d training examples!' % (X_multi_norm.shape[1]))
```
### Performance of the Neural Network Model for Multiple Linear Regression
    Exercise 6
    Predict the output of the trained model.
    Complete the function predict().
    Instructions: Calculate the prediction Y_hat of the trained model on the dataset X.
```python
### START CODE HERE ### (~ 1 line of code)
parameters_multi = nn_model(X_multi_norm, Y_multi_norm, num_iterations=100, print_cost=True)
### END CODE HERE ###

print("W = " + str(parameters_multi["W"]))
print("b = " + str(parameters_multi["b"]))

W_multi = parameters_multi["W"]
b_multi = parameters_multi["b"]

-- Normalize the dataset
X_pred_multi = np.array([[1710, 7], [1200, 6], [2200, 8]]).T

# Normalize using the same mean and standard deviation of the original training array X_multi.
X_multi_mean = np.array(np.mean(X_multi)).reshape((2,1))
X_multi_std = np.array(np.std(X_multi)).reshape((2,1))
X_pred_multi_norm = (X_pred_multi - X_multi_mean)/ X_multi_std
# Make predictions.
Y_pred_multi_norm = np.matmul(W_multi, X_pred_multi_norm) + b_multi
# Denormalize using the same mean and standard deviation of the original training array Y_multi.
Y_pred_multi = Y_pred_multi_norm * np.std(Y_multi) + np.mean(Y_multi)

print(f"Ground living area, square feet:\n{X_pred_multi[0]}")
print(f"Rates of the overall quality of material and finish, 1-10:\n{X_pred_multi[1]}")
print(f"Predictions of sales price, $:\n{np.round(Y_pred_multi)}")
```
# Week 4: Determinants and Eigenvectors

    - PCA: Principal Component Analysis: Dimentional redactionalgorithms Take your and try to capture the much 
      information about the data as possible.

    It reduces the dimensionality of the data while preserving as much of the data's
    Which is really good for storing model and PCA works with EigenValues and EigenVectors
  
### Determinant of the Product 

    - det(AB) = det(A)det(B)
    - The Product of a singular and non singular matrix is a singular matrix That is 
      because the determinant of a singular matrix is 0 and the non singular matrix is 
      different than 0 therefore the product of the two matrices is 0.
### Determinant of inverses
    - det(A^-1) = 1/det(A)
    - The determinant of the inverse of a matrix is equal to 1 divided by the determinant 
      of the matrix itself.
    - The determinant of a matrix is equal to the determinant of its transpose.
    - det(A^T) = det(A)
    - if determinant of det(A) = 5 then det(A^1) = 0.2 because 5^1 = 0.2
    - if det(B) = 0 then there is not inverse of B because 0^-1 is not defined.

    - det(A^-1) = 1/det(A)

```python
  Compute the determinant of the matrix A using numpy
import numpy as np

W = np.array([[1, 1, 0],
              [2, 0, 1],
              [-1, 1, 0]])

# Calculate the determinant of matrix W
determinant_W = np.linalg.det(W)
determinant_W
```
```python
  Compute the inverse of the matrix
import numpy as np

W = np.array([[1, 2, -1],
              [1, 0, 1],
              [0, 1, 0]])

# Calculate the determinant of matrix W
determinant_W = np.linalg.inv(W)
print(determinant_W)
```
```python
  Compute the inverse and the identity
  import numpy as np

W = np.array([[1, 2, -1],
              [1, 0, 1],
              [0, 1, 0]])

I = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# Calculate the determinant of matrix W
determinant_W = np.linalg.inv(W)
ID = np.matmul(W, I)
print(ID)
```

    Is the rank of the 3x3 Identity matrix (ID) singular or non-singular? 
    is non-singular. This is because:

    The determinant of the identity matrix is 1, which is non-zero. A non-zero determinant is a key property of a non-singular matrix.
    All rows (and columns) of the identity matrix are linearly independent, meaning none of the rows (or columns) can be written as a linear combination of the others. This linear independence is another characteristic of a non-singular matrix.
    The rank of the identity matrix is equal to its size (3 in this case), indicating it spans a 3-dimensional space, which aligns with the properties of a non-singular matrix.
    Therefore, the 3x3 identity matrix is non-singular.
```python
  Compute the linear transformation
import numpy as np

# Redefine the matrix W and the vector b
W = np.array([[1, 2, -1],
              [1, 0, 1],
              [0, 1, 0]])
b = np.array([5, -2, 0])

# Calculate the output result y_vec = W * b
y_vec = np.dot(W, b)

# Check if the transformation is singular or non-singular
# We do this by calculating the determinant of matrix W
det_W = np.linalg.det(W)

# Determining if the transformation is singular or non-singular
transformation_type = "non-singular" if det_W != 0 else "singular"

print(y_vec, transformation_type)
```
```python
  Extract the first and third columns of the matrix
  and compute their dot product
# Define the matrix Z
Z = np.array([[3, 5, 2],
              [1, 2, 2],
              [-7, 1, 0]])

# Extract the first and the third column of Z to consider them as vectors
vector_1 = Z[:, 0]  # First column
vector_3 = Z[:, 2]  # Third column

# Calculate the dot product of the two vectors
dot_product = np.dot(vector_1, vector_3)
dot_product
```
```python
  Find the mult of A and B, their inverse and the determinant of the inverse
import numpy as np
A = np.array([[5, 2, 3],
              [-1, -3, 2],
              [0, 1, -1]])

B = np.array([[1, 0, -4],
              [2, 1, 0],
              [8, -1, 0]])
mul = np.matmul(A, B)
print(mul)

invert = np.linalg.inv(mul)

inverse_determinant = np.linalg.det(invert)
print(inverse_determinant)
```
```python
We can even check to see if it is possible

# Calculate the determinant of the product matrix A * B
det_product_A_B = np.linalg.det(product_A_B)

# Calculate the determinant of the inverse of the product matrix A * B
# The determinant of the inverse is the inverse of the determinant
if det_product_A_B != 0:
    det_inverse_product_A_B = 1 / det_product_A_B
else:
    det_inverse_product_A_B = "Cannot be computed"

det_inverse_product_A_B
```
```python
  import numpy as np
A = np.array([[5, 2, 3],
              [-1, -3, 2],
              [0, 1, -1]])

B = np.array([[1, 0, -4],
              [2, 1, 0],
              [8, -1, 0]])
mul = np.matmul(A, B)
print(mul)

invert = np.linalg.inv(mul)

inverse_determinant = np.linalg.det(invert)
print(inverse_determinant)
```
### Eigenvalues and Eigenvectors
    -------- Basis -------------

    - Bases are a set of vectors that span a space. They are the linear combination of the 
      the basis.
    - If given two vectors v1 and v2 and a point any linear combination of v1 and
      v2 to reach that point represents a basis of that point.
 
    ---------- Non Basis -----------
 
    - Here given only one vector v1 and a point if the the one vector point
      to the exact location or direction of the point, then it can be consider 
      as a basis otherwise using that only direction we cannot reach the point.

    A basis is a minimum spaning set

    ----------- Span --------------

    - The span is a set of point that can be reach by working in the direction of these
      vectors in any combination.
    - Even if we have two vectors,if they are following the same direction then they are 
      not a basis.

    - Three vectors are two big to be a basis even thought we can reach
      any point in the space using these three vectors.

    - The length of the basis is called the dimension of the space.
   
### Eigen bases are very important for PCA
    Eigenbasis: An eigenbasis is a special basis that simplifies the representation of 
    a linear transformation. It consists of eigenvectors.

    Eigenvectors: Eigenvectors are special vectors that, when transformed
     by a matrix, are scaled (stretched or compressed) but maintain their direction. In other words, the transformation of an eigenvector results in a scaled version of itself, and its direction doesn't change.

    Eigenvalues: Eigenvalues are the scaling factors applied to eigenvectors
    during the linear transformation. Each eigenvector has an associated eigenvalue.

    Linear Transformation Example: The explanation provides an example of a 2x2 
    matrix and shows how it transforms two different bases. In the first basis (the fundamental basis), the transformation results in a parallelogram. In the second basis (an eigenbasis), the transformation also results in a parallelogram, but the sides of the parallelogram are parallel to the original basis vectors. This property makes the second basis an eigenbasis.

    Usefulness: Eigenbases are valuable because they simplify linear transformations.
    Instead of thinking about complex transformations, you can represent them as combinations of stretching along the directions defined by the eigenvectors.
      
### Example
     - Given A = [[2, 0], [0, 3]] and v = [1, 1]
     - To find the eigenvectors and eigenvalues of this matrix, we need to solve the characteristic equation:
    
       det(A - λI) = 0

    Where:
    A is the given matrix.
    λ is the eigenvalue we're trying to find.
    I is the identity matrix.

     A - λI = [[2-λ, 1], 
               [0, 3-λ]]

    - Calcul the determinant
      det(A - λI) = (2-λ)(3-λ) - (0 * 1) = (2 - λ) (3 - λ) = 0

    - Find the eigenvalues by setting the determinant equal to 0 and solving for λ:
      (2 - λ) (3 - λ) = 0

    This equation has two solutions:
      * When 2 - λ = 0, λ1 = 2
      * When 3 - λ = 0, λ2 = 3 
    Therefore the eigenvalues are 2 and 3.

      λ = 2, 3

    --------- Finding the corresponding eigenvectors ------------
    - for λ1 = 2
      Subtitute λ1 = 2 into the equation A - λI = 0
      A - λI = A - 2I = [[2-2, 1], 
                         [0, 3-2]] = [[0, 1], 
                                      [0, 1]]

      - Solve for the eigenvector associated with λ1 by solving the 
        equation (A - 2I)v1 = 0
        [[0, 1],   [[x],   [[0], 
         [0, 1]] * [y]]  =  [0]]
                    
    - This gives us the equation y = 0, which means that the eigenvector v1 = [1, 0] is associated with λ1 = 2.

    ----- For λ2 = 3
      Subtitute λ2 = 3 into the equation A - 3I = 0
      A - λI = A - 3I = [[2-3, 1], 
                         [0, 3-3]] = [[-1, 1], 
                                      [0, 0]]

      - Solve for the eigenvector associated with λ2 by solving the 
        equation (A - 3I)v2 = 0
        [[-1, 1],   [[x],   [[0], 
         [0, 0]] *  [y]]  =  [0]]

    - This equation x - y = 0, which meand that the eigenvector v2 = [1, 1]

```python

Find the eigenvalues and eigenvectors of the matrix A using numpy

import numpy as np

# Define the matrix A
A = np.array([[9, 4],
              [4, 3]])

# Calculate the eigenvalues
eigenvalues = np.linalg.eigvals(A)

print("Eigenvalues:", eigenvalues)


------- eigenvectors ------------
import numpy as np

# Define the matrix A
A = np.array([[9, 4],
              [4, 3]])

# Calculate the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigvals(A),  np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
```
```python
  to find the characteristic equation of the matrix A
from sympy import symbols, Matrix

# Define the variable and matrix
lambda_ = symbols('lambda')
A = Matrix([[2, -3], [1, 6]])

# Calculate the characteristic polynomial
char_poly = (A - lambda_ * Matrix.eye(2)).det()

char_poly
```

===============================================================================

---

### Calculus for machine learning and Data science

    - Linear regression problem
    - Classification problem => Sentiment analysis

    - Math concepts used in model training
       - Gradients
       - Derivatives
       - Optimization
       - Loss and cost functions
       - Gradient Descent


    - A derivatives is a continuous rate of change of a function.
    - Velocity = distance / time
    - Slope = rise / run
    - Slope = Variation of x / Variation of time
    - Slope = Δx / Δt  
    - Slope of a tangeant at a point is the derivative of the function at that point. dx/dt

