import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv("datas/archive/sign_mnist_train/sign_mnist_train.csv")
test = pd.read_csv("datas/archive/sign_mnist_test/sign_mnist_test.csv")

# Mise en place
y_train = train['label'].values
y_test = test['label'].values
X_train = train.drop(['label'], axis=1).values
X_test = test.drop(['label'], axis=1).values

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Normaliser les données
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Conversion des étiquettes
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Modèle
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y_train.argmax(axis=1))), activation='softmax'))


model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_train, y_train, batch_size=64, epochs=20, verbose=1, validation_data=(X_test, y_test))

# Évaluation du modèle
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (accuracy * 100))

# Rapport de classification
y_true = y_test.argmax(axis=1)
y_pred = model.predict(X_test).argmax(axis=1)
print(classification_report(y_true, y_pred))


cm = confusion_matrix(y_true, y_pred)

# Matrice de confusion
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');

# Calcul de la precision
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size = 15);
plt.show()

# Sauvegarde du modèle
model.save('model_keras.h5')

