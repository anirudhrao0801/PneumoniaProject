from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def create_cnn_model():
    cnn = Sequential()

    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Conv2D(32, (3, 3), activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Flatten())

    cnn.add(Dense(activation='relu', units=128))
    cnn.add(Dense(activation='sigmoid', units=1))

    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return cnn
