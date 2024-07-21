from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target
class_name = iris.target_names
print(x)
print(iris.values())

print(y)
print(class_name)

clf = DecisionTreeClassifier()
clf.fit(x, y)

plt.figure(figsize=(10, 8))
plot_tree(clf, filled=True, class_names=class_name, feature_names=iris.feature_names)
plt.title('Decision Tree classification')
plt.show()

