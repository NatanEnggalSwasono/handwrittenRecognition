import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical

# Simpan data gambar dan label dalam array
# Gantilah dengan cara memuat dan memproses dataset Anda sendiri
# Contoh: X = np.array([gambar1, gambar2, ...])
# Contoh: y = np.array([label1, label2, ...])

# Bagi dataset menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi nilai piksel ke rentang 0-1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Konversi label menjadi one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Definisikan model CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluasi model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')