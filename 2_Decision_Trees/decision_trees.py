#Prerequisites :
# I am using python3 and hence had to install the following to get this sample working
# pip install pyparsing==2.2.0
# pip install pydot => installs pydot-1.2.3
# Install pyparser first and then pydot

# Reference : https://www.youtube.com/watch?v=tNa99PG8hR8

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

# print (iris.feature_names)
# print (iris.target_names)
#
# for i in range(len(iris.target)):
#     print ("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))

test_idx = [0, 50, 100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print(clf.predict(test_data))
print(test_data)
print(test_target)

#visualization of decision tree
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                       out_file = dot_data,
                       feature_names = iris.feature_names,
                       class_names = iris.target_names,
                       filled = True,
                       rounded = False,
                       impurity = False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris_decision_tree.pdf")
