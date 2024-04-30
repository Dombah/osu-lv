import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt


#mreža ima 10 slojeva i 1,122,758 parametara
# točnost na testnom skupu: 72.01

# tocnost na testnom skupu nakon dropouta: 74.96 

# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train, dtype ="uint8")
y_test = to_categorical(y_test, dtype ="uint8")

# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva

# pri treniranju s 15 epoha se zaustavilo na 13. epohi 
my_callbacks = [
    keras.callbacks.EarlyStopping ( monitor ="val_loss" ,
        patience = 5 ,
        verbose = 1),
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn_dropout',
                                update_freq = 100)
]

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# 74.15 na 5 epoha
# Veliki batch size (90) tocnost: 73.24
model.fit(X_train_n,
            y_train,
            epochs = 5,
            batch_size = 10,
            callbacks = my_callbacks,
            validation_split = 0.1)


score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')

#4. zad
# 1). Kada je preveliki batch size moze doci do overfittinga, odnosno opada tocnost, kada je premala onda moze biti underfitting
# 2). S malom stopom ucenja se model trenira duze dog s velikom stopom ucenja krace. 
# 3). Ako izbacimo dovoljno dobar broj slojeva mozemo dobiti precizniji model. Npr ako je model pretreniran onda je to dobro. Ako izbacimo previse onda model postaje nepecizan
# 4). Smanjuje se preciznost modela

