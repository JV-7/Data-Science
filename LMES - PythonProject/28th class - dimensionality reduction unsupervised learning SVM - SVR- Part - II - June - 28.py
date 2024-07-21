import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)
''' x_train is fit_transform whereas x_test is only made transform not fit because to prevent from the 
 over fitting '''

svm_classifier = SVC(kernel='linear')

''' 
other options in kernel:
1. ploy
2. rbf
3. sigmoid

this is to make the decision surface smooth

gamma = 'scale'(rbf.poly sigmoid)
define how far the influence of the single training data

low value of gamma means 'FAR'
High value of gamma means 'Close'

degree (poly)
'''

svm_classifier.fit(x_train, y_train)

y_predication = svm_classifier.predict(x_test)

confusion_matrix = confusion_matrix(y_test, y_predication)
classification_report = classification_report(y_test, y_predication)

print(confusion_matrix)
print(classification_report)





