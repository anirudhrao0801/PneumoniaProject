import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def prepare_data(base_dir='../input/chest_xray/chest_xray'):
    train_folder = os.path.join(base_dir, 'train')
    val_folder = os.path.join(base_dir, 'val')
    test_folder = os.path.join(base_dir, 'test')

    train_n = os.path.join(train_folder, 'NORMAL')
    train_p = os.path.join(train_folder, 'PNEUMONIA')

    print(f"Number of normal training images: {len(os.listdir(train_n))}")

    # Display random normal and pneumonia images
    display_random_images(train_n, train_p)

    return create_data_generators(base_dir)

def display_random_images(normal_dir, pneumonia_dir):
    rand_norm = np.random.randint(0, len(os.listdir(normal_dir)))
    norm_pic = os.listdir(normal_dir)[rand_norm]
    norm_pic_address = os.path.join(normal_dir, norm_pic)
    print('Normal picture title:', norm_pic)

    rand_p = np.random.randint(0, len(os.listdir(pneumonia_dir)))
    sic_pic = os.listdir(pneumonia_dir)[rand_p]
    sic_address = os.path.join(pneumonia_dir, sic_pic)
    print('Pneumonia picture title:', sic_pic)

    norm_load = Image.open(norm_pic_address)
    sic_load = Image.open(sic_address)

    f = plt.figure(figsize=(10, 6))
    a1 = f.add_subplot(1, 2, 1)
    plt.imshow(norm_load)
    a1.set_title('Normal')

    a2 = f.add_subplot(1, 2, 2)
    plt.imshow(sic_load)
    a2.set_title('Pneumonia')
    plt.show()

def create_data_generators(base_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'val'),
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )

    test_set = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )

    return training_set, validation_generator, test_set
