# Program to distinguish apples and oranges
# features = [weight (g), texture (smooth[1]/bumpty[2])]
# labels = apple[1], orange[2]
# Reference : https://www.youtube.com/watch?v=cKxRvEZd3Mw
from sklearn import tree
features = [[140,1], [130,1],[150,2],[170,2]]
labels = [1, 1, 2, 2]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print (clf.predict([[160, 2]]))
