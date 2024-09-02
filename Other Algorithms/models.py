from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Dense, Conv2D, GlobalAveragePooling2D
from keras import optimizers
from keras.applications import VGG16, VGG19, ResNet50, DenseNet121

class models:
    @staticmethod
    def DenseClassifier(hidden_layer_sizes, nn_params, dropout=1):
        model = Sequential()
        model.add(Flatten(input_shape=nn_params['input_shape']))
        for ilayer in hidden_layer_sizes:
            model.add(Dense(ilayer, activation='relu'))
            if dropout:
                model.add(Dropout(dropout))
        model.add(Dense(units=nn_params['output_neurons'], activation=nn_params['output_activation']))
        model.compile(loss=nn_params['loss'],
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.95),
                      metrics=['accuracy'])
        return model

    @staticmethod
    def CNNClassifier(num_hidden_layers, nn_params, dropout=1):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=nn_params['input_shape'], padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        for _ in range(num_hidden_layers-1):
            model.add(Conv2D(32, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=nn_params['output_neurons'], activation=nn_params['output_activation']))

        opt = optimizers.RMSprop(lr=1e-4, decay=1e-6)
        model.compile(loss=nn_params['loss'],
                      optimizer=opt,
                      metrics=['accuracy'])
        return model

    @staticmethod
    def TransferClassifier(name, nn_params, trainable=True):
        expert_dict = {
            'VGG16': VGG16,
            'VGG19': VGG19,
            'ResNet50': ResNet50,
            'DenseNet121': DenseNet121
        }

        expert_conv = expert_dict[name](weights='imagenet',
                                        include_top=False,
                                        input_shape=nn_params['input_shape'])
        for layer in expert_conv.layers:
            layer.trainable = trainable

        expert_model = Sequential()
        expert_model.add(expert_conv)
        expert_model.add(GlobalAveragePooling2D())
        expert_model.add(Dense(128, activation='relu'))
        expert_model.add(Dropout(0.3))
        expert_model.add(Dense(64, activation='relu'))
        expert_model.add(Dense(nn_params['output_neurons'], activation=nn_params['output_activation']))

        expert_model.compile(loss=nn_params['loss'],
                             optimizer=optimizers.SGD(lr=1e-4, momentum=0.95),
                             metrics=['accuracy'])
        return expert_model
