from keras.models import *
from keras.layers import *
from models.segnet_layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

# SegNet
def segnet(input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_1 = Convolution2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_1)

    conv_2 = Convolution2D(128, (3, 3), padding="same", kernel_initializer='he_normal')(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)
    conv_2 = Convolution2D(128, (3, 3), padding="same", kernel_initializer='he_normal')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_2)

    conv_3 = Convolution2D(256, (3, 3), padding="same", kernel_initializer='he_normal')(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_3 = Convolution2D(256, (3, 3), padding="same", kernel_initializer='he_normal')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_3 = Convolution2D(256, (3, 3), padding="same", kernel_initializer='he_normal')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_3)

    conv_4 = Convolution2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(pool_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)
    conv_4 = Convolution2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)
    conv_4 = Convolution2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_4)

    conv_5 = Convolution2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(pool_4)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_5 = Convolution2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(conv_5)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_5 = Convolution2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(conv_5)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_5)

    unpool_1 = MaxUnpooling2D(size=(2, 2))([pool_5, mask_5])

    conv_6 = Convolution2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(unpool_1)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_6 = Convolution2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(conv_6)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_6 = Convolution2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(conv_6)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)

    unpool_2 = MaxUnpooling2D(size=(2, 2))([conv_6, mask_4])

    conv_7 = Convolution2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(unpool_2)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    conv_7 = Convolution2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(conv_7)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    conv_7 = Convolution2D(256, (3, 3), padding="same", kernel_initializer='he_normal')(conv_7)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    unpool_3 = MaxUnpooling2D(size=(2, 2))([conv_7, mask_3])

    conv_8 = Convolution2D(256, (3, 3), padding="same", kernel_initializer='he_normal')(unpool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_8 = Convolution2D(128, (3, 3), padding="same", kernel_initializer='he_normal')(conv_8)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_8 = Convolution2D(128, (3, 3), padding="same", kernel_initializer='he_normal')(conv_8)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)

    unpool_4 = MaxUnpooling2D(size=(2, 2))([conv_8, mask_2])

    conv_9 = Convolution2D(128, (3, 3), padding="same", kernel_initializer='he_normal')(unpool_4)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_9 = Convolution2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(conv_9)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)

    unpool_5 = MaxUnpooling2D(size=(2, 2))([conv_9, mask_1])

    conv_10 = Convolution2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(unpool_5)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)
    conv_10 = Convolution2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(conv_10)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)
    conv_10 = Convolution2D(num_classes, (1, 1), padding="same", kernel_initializer='he_normal')(conv_10)
    conv_10 = Activation('softmax')(conv_10)

    model = Model(inputs=inputs, outputs=conv_10)
    return model
