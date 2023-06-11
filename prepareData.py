import numpy as np
import os
import glob
import cv2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def train_generator(batch_size=32, imagesPath='data/images', imageFormat='.tif',
                    masksPath='data/masks', maskFormat='.tif', num_classes=1):
    image_names = glob.glob(imagesPath + '/*' + imageFormat)
    image_names.sort()
    image_names_subset = image_names[0:(len(os.listdir(path=imagesPath)))]
    images = [cv2.imread(image, 1) for image in image_names_subset]
    image_dataset = np.array(images)

    mask_names = glob.glob(masksPath + '/*' + maskFormat)
    mask_names.sort()
    mask_names_subset = mask_names[0:(len(os.listdir(path=masksPath)))]
    masks = [cv2.imread(mask, 0) for mask in mask_names_subset]
    mask_dataset = np.array(masks)

    labelencoder = LabelEncoder()
    n, h, w = mask_dataset.shape
    mask_dataset_reshaped = mask_dataset.reshape(-1, 1)
    mask_dataset_reshaped_encoded = labelencoder.fit_transform(mask_dataset_reshaped)
    mask_dataset_encoded = mask_dataset_reshaped_encoded.reshape(n, h, w)

    mask_dataset_encoded = np.expand_dims(mask_dataset_encoded, axis=3)
    image_dataset = image_dataset / 255.

    masks_cat = to_categorical(mask_dataset_encoded, num_classes=num_classes)
    masks = masks_cat.reshape((mask_dataset_encoded.shape[0], mask_dataset_encoded.shape[1],
                               mask_dataset_encoded.shape[2], num_classes))

    images = image_dataset

    data_gen_args = dict(rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         fill_mode="constant",
                         cval=0,
                         horizontal_flip=True,
                         vertical_flip=True,
                         zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    seed = 1
    image_generator = image_datagen.flow(images,
                                         batch_size=batch_size, seed=seed)

    mask_generator = mask_datagen.flow(masks,
                                       batch_size=batch_size, seed=seed)

    train_generator = zip(image_generator, mask_generator)
    for (imgs, masks) in train_generator:
        yield imgs, masks
