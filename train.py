import tensorflow as tf
from prepareData import train_generator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import *
from time import time


def dice_coefficient(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / (denominator + tf.keras.backend.epsilon())


def trainNN(model=None, epochs=100, modelName='unet-vgg16',
            imagesPath='data/images', valImagesPath='data/valImages', imageFormat='.tif',
            masksPath='data/masks', valMasksPath='data/valMasks', maskFormat='.tif',
            num_classes=1):
    dependencies = {
        'dice_coefficient': dice_coefficient
    }
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy',
                  metrics=['accuracy', dice_coefficient])
    callbacks = [
        ModelCheckpoint(f'{modelName}.h5', monitor='loss', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="loss", patience=5, factor=0.1, verbose=1),
        CSVLogger(f'{modelName}.csv'),
        EarlyStopping(monitor="loss", patience=10)
    ]

    start = time()
    model.fit(train_generator(batch_size=16, imagesPath=imagesPath, imageFormat=imageFormat,
                              masksPath=masksPath, maskFormat=maskFormat, num_classes=num_classes),
              steps_per_epoch=100,
              epochs=epochs,
              validation_data=train_generator(batch_size=16, imagesPath=valImagesPath, imageFormat=imageFormat,
                                              masksPath=valMasksPath, maskFormat=maskFormat, num_classes=num_classes),
              validation_steps=20,
              callbacks=callbacks)
    print(time() - start)
