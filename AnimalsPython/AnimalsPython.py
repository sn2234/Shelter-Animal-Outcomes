
import DataModel
from sklearn import cross_validation
from sklearn import neighbors
from sklearn.metrics import accuracy_score

dl = DataModel.DataLoader()

x, y = dl.loadTrainData("..\\train.csv")

(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(x, y, test_size=0.2)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)

acc_cv = accuracy_score(y_cv, knn.predict(x_cv))

print("Accuracy on CV dataset: {}".format(acc_cv))
