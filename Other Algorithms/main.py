import gdown
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint

from data_package import pkg
from helpers import helpers
from models import models

# File variables
image_data_url = 'https://drive.google.com/uc?id=1DNEiLAWguswhiLXGyVKsgHIRm1xZggt_'
metadata_url = 'https://drive.google.com/uc?id=1MW3_FU6qc0qT_uG4bzxhtEHy4Jd6dCWb'
image_data_path = './image_data.npy'
metadata_path = './metadata.csv'
image_shape = (64, 64, 3)

# Neural net parameters
nn_params = {
    'input_shape': image_shape,
    'output_neurons': 1,
    'loss': 'binary_crossentropy',
    'output_activation': 'sigmoid'
}

# Download data
gdown.download(image_data_url, './image_data.npy', True)
gdown.download(metadata_url, './metadata.csv', True)

# Pre-loading all data of interest
_all_data = np.load('image_data.npy')
_metadata = pkg.get_metadata(metadata_path, ['train', 'test', 'field'])

# Preparing definitions
get_metadata = lambda: pkg.get_metadata(metadata_path, ['train', 'test'])
get_train_data = lambda flatten=False: pkg.get_train_data(flatten=flatten, all_data=_all_data, metadata=_metadata, image_shape=image_shape)
get_test_data = lambda flatten=False: pkg.get_test_data(flatten=flatten, all_data=_all_data, metadata=_metadata, image_shape=image_shape)
get_field_data = lambda flatten=False: pkg.get_field_data(flatten=flatten, all_data=_all_data, metadata=_metadata, image_shape=image_shape)

plot_one_image = lambda data, labels=[], index=None: helpers.plot_one_image(data=data, labels=labels, index=index, image_shape=image_shape)
plot_acc = lambda history: helpers.plot_acc(history)

model_to_string = lambda model: helpers.model_to_string(model)
get_misclassified_data = helpers.get_misclassified_data
combine_data = helpers.combine_data

DenseClassifier = lambda hidden_layer_sizes: models.DenseClassifier(hidden_layer_sizes=hidden_layer_sizes, nn_params=nn_params)
CNNClassifier = lambda num_hidden_layers: models.CNNClassifier(num_hidden_layers, nn_params=nn_params)
TransferClassifier = lambda name: models.TransferClassifier(name=name, nn_params=nn_params)

monitor = ModelCheckpoint('./model.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Running of algorithms
metadata = get_metadata()
(train_data, train_labels) = get_train_data(flatten=True)
(test_data, test_labels) = get_test_data(flatten=True)

knn = KNeighborsClassifier(n_neighbors=5)
log = LogisticRegression()
dt = DecisionTreeClassifier(max_depth=2)

knn.fit(train_data, train_labels)
log.fit(train_data, train_labels)
dt.fit(train_data, train_labels)

predictions = knn.predict(test_data)
score = accuracy_score(test_labels, predictions)
print(score)

def score_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    score = accuracy_score(test_labels, predictions)
    print(score)

print(knn.score(test_data, test_labels))

print(f'KNN score: {knn.score(test_data, test_labels):.2f}')
print(f'Log score: {log.score(test_data, test_labels):.2f}')
print(f'DT score: {dt.score(test_data, test_labels):.2f}')
