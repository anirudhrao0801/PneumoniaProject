import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

class helpers:
    @staticmethod
    def plot_one_image(data, labels=[], index=None, image_shape=[64,64,3]):
        num_dims = len(data.shape)
        num_labels = len(labels)

        if num_dims == 1:
            data = data.reshape(image_shape)
        if num_dims == 2:
            data = data.reshape(np.vstack([-1, image_shape]))
        num_dims = len(data.shape)

        if num_dims == 3:
            if num_labels > 1:
                print('Multiple labels does not make sense for single image.')
                return
            label = labels[0] if labels else ''
            image = data
        elif num_dims == 4:
            image = data[index, :]
            label = labels[index] if labels else ''

        print(f'Label: {label}')
        plt.imshow(image)
        plt.show()

    @staticmethod
    def get_misclassified_data(data, labels, predictions):
        missed_index = np.where(np.abs(predictions.squeeze() - labels.squeeze()) > 0)[0]
        missed_labels = labels[missed_index]
        missed_data = data[missed_index,:]
        predicted_labels = predictions[missed_index]
        return missed_data, missed_labels, predicted_labels, missed_index

    @staticmethod
    def combine_data(data_list, labels_list):
        return np.concatenate(data_list, axis=0), np.concatenate(labels_list, axis=0)

    @staticmethod
    def model_to_string(model):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        sms = "\n".join(stringlist)
        sms = re.sub('_\d\d\d','', sms)
        sms = re.sub('_\d\d','', sms)
        sms = re.sub('_\d','', sms)
        return sms

    @staticmethod
    def plot_acc(history, ax=None, xlabel='Epoch #'):
        history = history.history
        history.update({'epoch': list(range(len(history['val_acc'])))})
        history = pd.DataFrame.from_dict(history)

        best_epoch = history.sort_values(by='val_acc', ascending=False).iloc[0]['epoch']

        if not ax:
            f, ax = plt.subplots(1,1)
        sns.lineplot(x='epoch', y='val_acc', data=history, label='Validation', ax=ax)
        sns.lineplot(x='epoch', y='acc', data=history, label='Training', ax=ax)
        ax.axhline(0.5, linestyle='--', color='red', label='Chance')
        ax.axvline(x=best_epoch, linestyle='--', color='green', label='Best Epoch')
        ax.legend(loc=1)
        ax.set_ylim([0.4, 1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Accuracy (Fraction)')
        plt.show()
