import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

axes1: Axes
axes2: Axes
fig: Figure

'''Below code is use for one diagram'''
# x = [1, 2, 3, 4, 5]

# y = [3, 5, 9, 12, 10]

# plt.plot(x, y)
# plt.show()

'''Below code is used for multiple diagram'''

# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [9, 5, 3, 7, 9, 3, 4, 6, 3, 6]

# x1 = [5, 10, 15, 20, 25]
# y1 = [15, 20, 30, 50, 65]

''' fig variable is used to set common attributes to both the plots '''
# fig, (axes1, axes2) = plt.subplots(1, 2)  # subplots - function (1, 2) indicates that 1 rows and 2 columns
# fig.subtitle('Two tables')
# fig.tight_layout()

# axes1.plot(x, y)
# axes1.set_title('Graph 1')
# axes1.set_xlabel('x label')
# axes1.set_ylabel('y label')

# axes2.plot(x1, y1, marker='x', linestyle='--')

# plt.show()

''' Below code is to show bar plot in figure '''

# x = ['Dhoni', 'Ashwin', 'Kohli', 'Raina']

# y = [7, 9, 13, 3]

# plt.bar(x, y)
# plt.show()

''' Below is to show histogram chart but with data generated from numpy'''

# data = np.random.randn(100)  # generate random 100 number that has mean = 0, std = 1
# plt.hist(data, edgecolor='Black', bins=50, color='Yellow')
# print(data.ndim)
# plt.show()
# print(data)

''' Scatter chart '''

# x = np.random.randn(100)
# y = 2*x + np.random.randn(100)
#
# plt.scatter(x, y)
#
# plt.show()

''' Pie Chart '''

# labels = ('Dhoni', 'Ashwin', 'Raina', 'Karthik')
# age = [42, 36, 38, 37]
# col = ['blue', 'green', 'yellow', 'gold']
# explode = [0, 0, 0.2, 0]
#
# plt.pie(age, explode=explode, colors=col, labels=labels, autopct='%1.2f%%')

# plt.show()

# x = [1, 2, 3, 4, 5, 6, 7]
# y = [2, 5, 8, 6, 3, 4, 6]

# plt.plot(x, y)

# plt.show()


