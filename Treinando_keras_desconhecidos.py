import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt


df_desconhecidos = pd.read_csv("faces_desconhecidos.csv")
print(df_desconhecidos.columns)
df_conhecidos = pd.read_csv("caio.csv")
df = pd.concat([df_desconhecidos, df_conhecidos])
df = df.drop(columns=['Unnamed: 0']).copy()
print(df.columns)
print(df_desconhecidos.head())
print(df_conhecidos.head())
print(df.head())
X = np.array(df.drop("target", axis=1))
y = np.array(df.target)

X, y = shuffle(X, y, random_state=0)

trainX, valX, trainY, valY = train_test_split(X, y, test_size=0.20, random_state=42)

norm = Normalizer(norm="l2")
trainX = norm.transform(trainX)
valX = norm.transform(valX)

print(np.unique(trainY))

out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
print(np.unique(trainY))
classes = len(np.unique(trainY))

out_encoder = LabelEncoder()
out_encoder.fit(valY)
valY = out_encoder.transform(valY)
print(np.unique(valY))

trainY = to_categorical(trainY)
valY = to_categorical(valY)

print(valY[0])
print(trainY[0])

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(128,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(classes, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

batch_size=8
epochs= 40

history = model.fit(trainX, trainY,
                    epochs=epochs,
                    validation_data = (valX,valY),
                    batch_size=batch_size)
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

yhat_val = model.predict(valX)

valY = np.argmax(valY,axis = 1)
yhat_val = np.argmax(yhat_val,axis = 1)

print(valY[0])
print(yhat_val[0])

from sklearn.metrics import confusion_matrix


def print_confusion_matrix(model_name, valY, yhat_val):
    cm = confusion_matrix(valY, yhat_val)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    print("MODELO : {}".format(model_name))
    print("Acurácia: {:.4f}".format(acc))
    print("Sensitividade: {:.4f}".format(sensitivity))
    print("Especificidade: {:.4f}".format(specificity))

    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(5, 5))
    plt.show()

print_confusion_matrix("KERAS", valY, yhat_val)

model.save("faces_d.h5")