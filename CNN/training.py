import matplotlib.pyplot as plt

def train_model(model, training_set, validation_generator):
    cnn_model = model.fit_generator(
        training_set,
        steps_per_epoch=163,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=624
    )
    return cnn_model

def evaluate_model(model, test_set):
    test_accu = model.evaluate_generator(test_set, steps=624)
    print('The testing accuracy is:', test_accu[1]*100, '%')

def plot_accuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper left')
    plt.show()

def plot_loss(history):
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Test set'], loc='upper left')
    plt.show()
