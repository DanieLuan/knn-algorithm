from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import myKnn

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)

myKnn = myKnn.knn(k=5, distance_metric='manhattan')
myKnn.fit(X_train, y_train)
y_pred = myKnn.predict(X_test)

knn_sklearn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_sklearn.fit(X_train, y_train)
y_predSk = knn_sklearn.predict(X_test)

print("MY KNN ACCURACY: ", metrics.accuracy_score(y_test, y_pred))

print("SKLEARN KNN ACCURACY: ", metrics.accuracy_score(y_test, y_predSk))

report_sklearn = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report_sklearn)

report_mine = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report_mine)