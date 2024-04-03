import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay, classification_report

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

colors = ListedColormap([[1,0,1], [1,0,0]])

plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], c = y_train, marker='o', label='Train data', cmap = colors)
plt.scatter(X_test[:,0], X_test[:,1], c = y_test ,marker='x', label='Test data', cmap = colors)
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.title('Scatter plot of training data and test data')



LogRegression_model = LogisticRegression ()
LogRegression_model.fit (X_train, y_train)

print(LogRegression_model.coef_, LogRegression_model.intercept_)

a = -(LogRegression_model.coef_[0][0]) / (LogRegression_model.coef_[0][1])
b = -(LogRegression_model.intercept_[0]) / (LogRegression_model.coef_[0][1])


x_values = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)

y_values = a * x_values + b

plt.plot(x_values, y_values, color='black', label='Decision Boundary')
plt.fill_between(x_values, y_values, np.max(X_train[:, 1]), color='skyblue', alpha=0.3)
plt.fill_between(x_values, y_values, np.min(X_train[:, 1]), color='salmon', alpha=0.3)
plt.show()

y_test_p = LogRegression_model.predict (X_test)

cm = confusion_matrix (y_test , y_test_p)
print (" Matrica zabune : " , cm)
disp = ConfusionMatrixDisplay ( confusion_matrix (y_test , y_test_p ))
disp.plot()

plt.show()

print (classification_report(y_test , y_test_p))

correctly_predicted = X_test[y_test == y_test_p]
incorrectly_predicted = X_test[y_test != y_test_p]

plt.scatter(correctly_predicted[:,0], correctly_predicted[:,1], color = 'green')
plt.scatter(incorrectly_predicted[:,0], incorrectly_predicted[:,1], color = 'black')
plt.show()