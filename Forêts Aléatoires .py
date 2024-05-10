from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.models, random_state=42)


forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)

accuracy = forest.score(X_test, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')
