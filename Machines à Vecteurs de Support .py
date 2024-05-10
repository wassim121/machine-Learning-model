from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris.data[:, :2]  #
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


svm = SVC(kernel='linear')
svm.fit(X_train, y_train)


plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter')
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')
plt.title('SVM sur Iris Dataset')
plt.show()
