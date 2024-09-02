import pandas as pd
import numpy as np

class pkg:
    @staticmethod
    def get_metadata(metadata_path, which_splits=['train', 'test']):
        metadata = pd.read_csv(metadata_path)
        keep_idx = metadata['split'].isin(which_splits)
        return metadata[keep_idx]

    @staticmethod
    def get_data_split(split_name, flatten, all_data, metadata, image_shape):
        sub_df = metadata[metadata['split'].isin([split_name])]
        index = sub_df['index'].values
        labels = sub_df['class'].values
        data = all_data[index,:]
        if flatten:
            data = data.reshape([-1, np.product(image_shape)])
        return data, labels

    @staticmethod
    def get_train_data(flatten, all_data, metadata, image_shape):
        return pkg.get_data_split('train', flatten, all_data, metadata, image_shape)

    @staticmethod
    def get_test_data(flatten, all_data, metadata, image_shape):
        return pkg.get_data_split('test', flatten, all_data, metadata, image_shape)

    @staticmethod
    def get_field_data(flatten, all_data, metadata, image_shape):
        return pkg.get_data_split('field', flatten, all_data, metadata, image_shape)
