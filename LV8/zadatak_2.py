import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

num_classes = 10
input_shape = (28, 28, 1)

model = keras.saving.load_model('model.h5')
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

x_train_s =np.reshape(x_train_s,(60000,784))
x_test_s =np.reshape(x_test_s,(10000,784))


# predicit from model
predictions = model.predict(x_test_s)
predictions_class = np.argmax(predictions, axis=1)
y_test_class = np.argmax(y_test_s, axis=1)


missclassified = np.where(predictions_class != y_test_class)[0]
plt.subplots(3,3, figsize=[12,12])
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(x_test[missclassified[i]], cmap='gray', interpolation='none')
    plt.title("Predicted: {}, Actual: {}".format(predictions_class[missclassified[i]], y_test_class[missclassified[i]]))
    plt.xticks([])
    plt.yticks([])
plt.show()


img = plt.imread('sedmica.png')
img = np.resize(img, (1, 784))
plt.imshow(img, cmap='gray')
plt.show()



