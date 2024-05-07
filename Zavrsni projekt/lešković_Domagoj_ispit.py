#import bibilioteka
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, GridSearchCV
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)

##################################################
#1. zadatak
##################################################

#učitavanje dataseta

titanic = pd.read_csv("titanic.csv")
females = titanic[titanic['Sex'] == 'female']
#a)
print(len(females))
#b)

died = titanic[titanic['Survived'] == 0]
percent = len(died) / len(titanic) * 100
print(f'Postotak umrlih: {round(percent,2)}%')
#c)

males = titanic[titanic['Sex'] == 'male']
maleSurvivors = males[males['Survived'] == 1]
malesPercent = round(len(maleSurvivors) / len(males) * 100, 2)

femaleSurvivors = females[females['Survived'] == 1]
femalePercent = round(len(femaleSurvivors) / len(females) * 100, 2)

x = ['Male percent', 'Female percent']
y = [malesPercent,femalePercent]
colors = ['green', 'yellow']
plt.figure()
plt.bar(x,y, color = colors)
plt.xlabel('Spol')
plt.ylabel('Postotak prezivjelih')
plt.title('Dijagram prezivjelih po spolu u postotcima')
plt.show()
#d)
print(maleSurvivors['Age'].mean(axis=0))
print(femaleSurvivors['Age'].mean(axis=0))

#e)
classes = maleSurvivors['Pclass'].dropna()
print(classes.drop_duplicates())

class1Males = maleSurvivors[maleSurvivors['Pclass'] == 1]
class2Males = maleSurvivors[maleSurvivors['Pclass'] == 2]
class3Males = maleSurvivors[maleSurvivors['Pclass'] == 3]

print(class1Males['Age'].max())
print(class2Males['Age'].max())
print(class3Males['Age'].max())

##################################################
#2. zadatak
##################################################

features = ['Pclass', 'Sex', 'Fare', 'Embarked']
titanic_copy = titanic.copy()
titanic_copy = titanic_copy.dropna()
X = pd.get_dummies(titanic_copy)
y = titanic_copy['Survived'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, random_state = 10)

scaler = StandardScaler()
X_train_n = scaler.fit_transform (X_train)
X_test_n = scaler.transform ((X_test))

#učitavanje dataseta

#train test split

#a)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_n,y_train)
#b)
y_train_p = model.predict(X_train_n)
y_test_p = model.predict(X_test_n)

print("KNN regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

scores = cross_val_score(model , X_train , y_train , cv =5)
print ( scores )
#c)
param_grid = {'n_neighbors': np.arange(1, 30)}
KNN_model_2 = KNeighborsClassifier ()
knn_gscv = GridSearchCV(KNN_model_2, param_grid, cv=5)
knn_gscv.fit(X_train_n, y_train)

print ( 'Best param from 1 to 30: ' , knn_gscv.best_params_ )
model = KNeighborsClassifier(n_neighbors= 16)
model.fit(X_train_n,y_train)

#d)
y_train_p = model.predict(X_train_n)
y_test_p = model.predict(X_test_n)

print("KNN regresija s optimalnim k: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
##################################################
#3. zadatak
##################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state = 10)
#učitavanje podataka:

#a)
print(X)
model = keras.Sequential()
model.add(layers.Input(shape=(455,)))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
#b)
model.compile (loss ="binary_crossentropy" ,optimizer ="adam", metrics =["accuracy",])
#c)
#Komentirano jer je treniran model spremljen u datoteku
'''
batch_size = 5
epochs = 100
history = model.fit ( X_train ,y_train ,batch_size = batch_size ,epochs = epochs ,validation_split = 0.25)
#d)
model.save('model.h5')
del model
'''
model = keras.saving.load_model('model.h5')
#e)
predictions = model.predict ( X_test )
score = model.evaluate ( X_test , y_test , verbose =0)
print('Test accuracy:', score[1])
#f)
y_pred = np.argmax(predictions, axis=1)
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = np.arange(2))

cm_display.plot()
plt.show()