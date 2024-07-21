""" NumPy """

''' To begin numpy is the multi dimensional  array library so what that mean numpy can store data in
 D - Dimensional
 1D array - Shape(4,)  # please screenshot 1
 2D array - Shape(2,3) # please screenshot 1
 3D array - Shape(4,3,2) # please screenshot 1
 
 Why is Numpy Faster? - Fixed Type
 
 >>Faster to read with less bytes of memory
 >>No type checking when iterating though objects
 >>Contiguous Memory
 
 Application of Numpy:
 
 >> Mathematics (MATLAB Replacement)
 >> Plotting (Matplotlib)
 >> Machine Learning
 
 '''

import numpy as np

a = np.array([1, 2, 3])
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f' ')
print(f'''
Array a: \n{a}\n
Array b: \n{b}\n
Dimension a: {a.ndim}
Dimension b: {b.ndim}\n
Type a: {a.dtype}
Type b: {b.dtype}\n
Shape a: {a.shape}
Shape b: {b.shape}\n
Total bytes(dtype) a: {a.nbytes}
Total bytes(dtype) b: {b.nbytes}\n'''
      )

''' we can also define the dtype - means data type '''
c = np.array([22, 66, 45], dtype='int16')

''' Accessing/Changing specific elements, rows, columns, etc '''

Array_1 = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
''' to get the specific values in the above array - [row, column] '''

print(f'''
Array_1: \n {Array_1}\n
specific values 2nd row of 5th value: {Array_1[0, 5]}

Specific row - 2: \n {Array_1[1, :]}
Specific column 5: \n {Array_1[:, 4]}

Between the number: \n {Array_1[0, 1:5]}
Between the number: \n {Array_1[0, 1:5:2]}        # '2' - is the step element
''')

'''------------------------- Changing the values in the specific rows and columns ----------------------------------'''

Array_1[1, 5] = 100
print(f' 1. updated Array \n{Array_1}')

Array_1 = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
Array_1[0, :] = 505
print(f''' changed the entire row
      2. updated Array \n{Array_1}''')

Array_1 = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
Array_1[:, 1] = 505
print(f''' changed the entire colum
      3. updated Array \n{Array_1}''')

Array_1 = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
Array_1[:, 1] = [505, 100]
print(f''' changed the entire colum
      4. updated Array \n{Array_1}''')

print()

print('----------------Initializing Different Array Types------------------')
'''--------------------------Initializing Different Array Types-------------------------'''
print()
print(f''' 
All as 0 Matrix \n{np.zeros((3, 5))}  # 3 dimension and 2x5 matrix \n
All as 1 matrix \n{np.ones((4, 2, 5))}  # 4 dimension and 2x5 matrix
''')
