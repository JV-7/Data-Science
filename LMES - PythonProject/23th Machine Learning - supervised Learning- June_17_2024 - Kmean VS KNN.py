import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

x_train = np.array([
    [25, 50000],
    [30, 60000],
    [35, 70000],
    [40, 80000],
    [45, 90000],
    [50, 100000],
    [55, 110000],
    [60, 1200000]
])

# Labels: 0 - Low - Income and 1 - High - Income Group
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# New data points to classify
x_test = np.array([
    [28, 55000],
    [48, 95000],
    [33, 64000],
    [53, 305000]
])

# True labels for the test data
y_test = np.array([0, 1, 0, 1])

# create a K-NN classifier with k = 3
KNN = KNeighborsClassifier(n_neighbors=3)

# Fit the K-NN classifier to the training data
KNN.fit(x_train, y_train)

# predict the labels for the test data
y_pred = KNN.predict(x_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'''
Predicated Labels for test data: {y_pred} \n
Accuracy: {accuracy} \n
Confusion Matrix: \n
{conf_matrix} \n
Classification Report: 
{class_report} \n
''')

# Plotting the data and the decision boundaries

plt.figure(figsize=(10, 6))

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='o', label='Training data')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x', label='Test Data')

# Label the axes

plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.title('K-NN Classification')
plt.show()






























































