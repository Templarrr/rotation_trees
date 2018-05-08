from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from model.rotation_tree import RotationTreeClassifier, RotationTreeRegressor

boston_data = datasets.load_boston()
clf = RotationTreeRegressor()
clf.fit(boston_data.data, boston_data.target)
print cross_val_score(clf,
                      boston_data.data,
                      boston_data.target,
                      scoring='neg_mean_absolute_error',
                      cv=10)
clf2 = DecisionTreeRegressor()
print cross_val_score(clf2,
                      boston_data.data,
                      boston_data.target,
                      scoring='neg_mean_absolute_error',
                      cv=10)

cancer_data = datasets.load_breast_cancer()
clf = RotationTreeClassifier()
print cross_val_score(clf,
                      cancer_data.data,
                      cancer_data.target,
                      scoring='f1',
                      cv=5)
clf2 = DecisionTreeClassifier()
print cross_val_score(clf2,
                      cancer_data.data,
                      cancer_data.target,
                      scoring='f1',
                      cv=5)
