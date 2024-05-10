from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


X = 10 * np.random.rand(100, 1)
y = 3 * X + np.random.randn(100, 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
plt.scatter(X_test, y_test, color='blue', label='Données réelles')
plt.plot(X_test, y_pred, color='red', label='Prédiction linéaire')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Régression Linéaire')
plt.legend()
plt.show()
